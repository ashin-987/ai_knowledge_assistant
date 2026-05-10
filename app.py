"""
Improved Streamlit Web Interface
Features: Streaming, feedback, analytics, document preview, export
"""

import streamlit as st
import os
from pathlib import Path
import shutil
import json
from datetime import datetime
import time

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine

# Page configuration
st.set_page_config(
    page_title="AI Knowledge Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main improvements */
    .stChatMessage {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 0.8rem;
    }
    
    /* Feedback buttons */
    .feedback-container {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    
    /* Source badges */
    .source-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background-color: #f0f2f6;
        border-radius: 0.4rem;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    
    /* Metrics */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Status indicators */
    .status-success {
        color: #00cc00;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ff9900;
        font-weight: bold;
    }
    
    /* Suggested questions */
    .suggested-question {
        padding: 0.8rem;
        background-color: #f8f9fa;
        border-left: 3px solid #4CAF50;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        cursor: pointer;
    }
    
    .suggested-question:hover {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None

if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "llama-3.1-8b-instant"

if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {}

if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []

# Check for API token
from dotenv import load_dotenv
load_dotenv()

def check_api_token():
    """Check if Groq API key is configured"""
    try:
        token = st.secrets["GROQ_API_KEY"]
        if token and token.startswith("gsk_"):
            return True
    except:
        pass
    
    token = os.getenv("GROQ_API_KEY", "")
    return bool(token and token.startswith("gsk_"))

has_token = check_api_token()

# Title with status
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🤖 AI Knowledge Assistant Pro")
    st.markdown("**Advanced RAG System with Hybrid Search & Reranking**")

with col2:
    if has_token:
        st.markdown('<p class="status-success">✅ API Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-warning">⚠️ Setup Required</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API status
    with st.expander("🔑 API Configuration", expanded=not has_token):
        if has_token:
            st.success("✅ Groq API Key: Configured")
        else:
            st.error("❌ Groq API Key: Missing")
            st.info("""
            **Setup Instructions:**
            1. Get free API key: https://console.groq.com/keys
            2. Add to Streamlit Cloud secrets:
               `GROQ_API_KEY = "your_key_here"`
            3. Or add to local `.env` file
            """)
    
    st.divider()
    
    # Model Selection
    st.subheader("🤖 AI Model")
    
    model_options = {
        model_id: f"{info['name']} ({info['speed']})" 
        for model_id, info in RAGEngine.AVAILABLE_MODELS.items()
    }
    
    default_model = "llama-3.1-8b-instant"
    default_index = list(model_options.keys()).index(default_model)
    
    selected_model = st.selectbox(
        "Choose Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=default_index,
        help="Different models offer different speed/quality tradeoffs"
    )
    
    # Model info
    if selected_model in RAGEngine.AVAILABLE_MODELS:
        model_info = RAGEngine.AVAILABLE_MODELS[selected_model]
        with st.expander("ℹ️ Model Details"):
            st.markdown(f"**{model_info['name']}**")
            st.caption(model_info['description'])
            st.caption(f"Max tokens: {model_info['max_tokens']}")
    
    st.divider()
    
    # Advanced Settings
    with st.expander("⚡ Advanced Settings"):
        use_rerank = st.checkbox("Use Reranking", value=True, 
                                help="Improves precision but slower")
        use_query_expansion = st.checkbox("Query Expansion", value=False,
                                         help="Generate query variations")
        n_results = st.slider("Results to retrieve", 3, 10, 5,
                            help="More results = better context but slower")
    
    st.divider()
    
    # Document Upload
    st.header("📁 Documents")
    
    uploaded_files = st.file_uploader(
        "Upload Files",
        type=['pdf', 'txt', 'docx', 'pptx'],
        accept_multiple_files=True,
        help="Supported: PDF, TXT, DOCX, PPTX"
    )
    
    # Document preview
    if uploaded_files:
        with st.expander(f"📄 Uploaded Files ({len(uploaded_files)})"):
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024  # KB
                st.markdown(f"- **{file.name}** ({file_size:.1f} KB)")
    
    # Process button
    if st.button("🔄 Process Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ Please upload files first")
        elif not has_token:
            st.error("❌ API token required. Configure in secrets/env.")
        else:
            with st.spinner("Processing documents..."):
                try:
                    # Save files
                    upload_dir = "./uploaded_documents"
                    if os.path.exists(upload_dir):
                        shutil.rmtree(upload_dir)
                    os.makedirs(upload_dir)
                    
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(upload_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Process with improved processor
                    start_time = time.time()
                    processor = DocumentProcessor()
                    documents = processor.process_directory(upload_dir)
                    processing_time = time.time() - start_time
                    
                    if documents:
                        # Create vector store with hybrid search
                        st.session_state.vector_store = VectorStore()
                        st.session_state.vector_store.add_documents(documents)
                        
                        # Initialize RAG engine
                        st.session_state.rag_engine = RAGEngine(
                            st.session_state.vector_store,
                            model_name=selected_model
                        )
                        st.session_state.selected_model = selected_model
                        st.session_state.documents_processed = True
                        
                        # Store stats
                        stats = st.session_state.vector_store.get_stats()
                        st.session_state.processing_stats = {
                            'files': len(uploaded_files),
                            'chunks': stats['total_chunks'],
                            'processing_time': processing_time,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Generate suggested questions
                        with st.spinner("Generating suggested questions..."):
                            suggestions = st.session_state.rag_engine.get_suggested_questions(3)
                            st.session_state.suggested_questions = suggestions
                        
                        st.success(f"✅ Processed {len(uploaded_files)} files ({stats['total_chunks']} chunks) in {processing_time:.1f}s")
                        st.balloons()
                    else:
                        st.error("❌ No content found in files")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
    
    # Stats and Controls (only if processed)
    if st.session_state.documents_processed:
        st.divider()
        st.subheader("📊 Database Stats")
        
        stats = st.session_state.vector_store.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", stats['total_chunks'])
            st.metric("Messages", len(st.session_state.messages))
        
        with col2:
            if st.session_state.processing_stats:
                st.metric("Files", st.session_state.processing_stats['files'])
                st.metric("Process Time", f"{st.session_state.processing_stats['processing_time']:.1f}s")
        
        st.caption(f"🤖 Model: {RAGEngine.AVAILABLE_MODELS[st.session_state.selected_model]['name']}")
        st.caption(f"🔍 Hybrid Search: BM25 + Semantic")
        st.caption(f"🎯 Reranking: {'Enabled' if use_rerank else 'Disabled'}")
        
        st.divider()
        
        # Export and Clear
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Chat", use_container_width=True):
                if st.session_state.messages:
                    chat_data = {
                        'timestamp': datetime.now().isoformat(),
                        'model': st.session_state.selected_model,
                        'stats': st.session_state.processing_stats,
                        'messages': st.session_state.messages,
                        'feedback': st.session_state.feedback_data
                    }
                    
                    json_str = json.dumps(chat_data, indent=2)
                    st.download_button(
                        label="💾 Download JSON",
                        data=json_str,
                        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No messages to export")
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.clear_history()
                st.rerun()
        
        if st.button("🔄 Reset Database", use_container_width=True):
            if st.session_state.vector_store:
                st.session_state.vector_store.reset()
            st.session_state.documents_processed = False
            st.session_state.messages = []
            st.session_state.suggested_questions = []
            st.rerun()
    
    # Help section
    st.divider()
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        **Quick Start:**
        1. Upload documents (PDF, TXT, DOCX, PPTX)
        2. Click "Process Documents"
        3. Ask questions in the chat!
        
        **Features:**
        - 🔍 Hybrid search (BM25 + Semantic)
        - 🎯 Neural reranking for precision
        - 💬 Conversation memory
        - 📊 Source attribution
        - 📥 Export chat history
        
        **Tips:**
        - Use specific questions for best results
        - Check sources to verify answers
        - Try suggested questions to explore
        """)
    
    st.divider()
    st.caption("💡 Powered by Groq, ChromaDB & Sentence Transformers")
    st.caption("🚀 Advanced RAG with Hybrid Search")

# Main chat area
if not st.session_state.documents_processed:
    # Welcome screen
    st.info("👈 **Get Started:** Upload documents in the sidebar and click 'Process Documents'")
    
    # Feature grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚀 Advanced RAG")
        st.markdown("Hybrid search combines keyword (BM25) and semantic matching for better results")
    
    with col2:
        st.markdown("### 🎯 High Precision")
        st.markdown("Neural reranking ensures the most relevant information is used")
    
    with col3:
        st.markdown("### 💬 Smart Chat")
        st.markdown("Conversation memory for contextual follow-up questions")
    
    # Model comparison
    st.divider()
    st.subheader("🤖 Available Models")
    
    for model_id, model_info in RAGEngine.AVAILABLE_MODELS.items():
        with st.expander(f"**{model_info['name']}** - {model_info['speed']}"):
            st.markdown(f"**Description:** {model_info['description']}")
            st.markdown(f"**Max Tokens:** {model_info['max_tokens']}")
            st.markdown(f"**Speed:** {model_info['speed']}")

else:
    # Suggested questions (show once at start)
    if st.session_state.suggested_questions and len(st.session_state.messages) == 0:
        st.subheader("💡 Suggested Questions")
        
        for i, question in enumerate(st.session_state.suggested_questions):
            if st.button(f"💭 {question}", key=f"suggest_{i}", use_container_width=True):
                # Trigger this question
                st.session_state.auto_query = question
                st.rerun()
        
        st.divider()
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                
                # Sources
                if metadata.get('sources'):
                    with st.expander(f"📎 Sources ({len(metadata['sources'])})"):
                        for source in metadata['sources']:
                            st.markdown(f"- 📄 **{source}**")
                        
                        if 'retrieval_scores' in metadata:
                            st.caption("**Top Results:**")
                            for score_info in metadata['retrieval_scores']:
                                st.caption(f"  • {score_info['source']}: {score_info['score']:.3f}")
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏱️ {metadata.get('response_time', 0):.2f}s")
                with col2:
                    st.caption(f"📄 {metadata.get('retrieved_chunks', 0)} chunks")
                with col3:
                    st.caption(f"🤖 {metadata.get('model', 'N/A')}")
                
                # Feedback buttons
                col1, col2, col3 = st.columns([1, 1, 10])
                
                feedback_key = f"feedback_{idx}"
                
                with col1:
                    if st.button("👍", key=f"up_{idx}"):
                        st.session_state.feedback_data.append({
                            'message_idx': idx,
                            'rating': 'positive',
                            'timestamp': datetime.now().isoformat()
                        })
                        st.success("Thanks!", icon="✅")
                
                with col2:
                    if st.button("👎", key=f"down_{idx}"):
                        st.session_state.feedback_data.append({
                            'message_idx': idx,
                            'rating': 'negative',
                            'timestamp': datetime.now().isoformat()
                        })
                        st.warning("Feedback noted", icon="📝")
    
    # Auto-query from suggested questions
    if hasattr(st.session_state, 'auto_query'):
        prompt = st.session_state.auto_query
        del st.session_state.auto_query
    else:
        # Chat input
        prompt = st.chat_input("Ask a question about your documents...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate answer with streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            metadata = {}
            try:
                generator = st.session_state.rag_engine.generate_answer_streaming(prompt)
                
                for chunk in generator:
                    if isinstance(chunk, str):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    else:
                        # Final metadata
                        metadata = chunk
                
                # Remove cursor
                response_placeholder.markdown(full_response)
                
                # Show sources immediately
                if metadata.get('sources'):
                    with st.expander(f"📎 Sources ({len(metadata['sources'])})"):
                        for source in metadata['sources']:
                            st.markdown(f"- 📄 **{source}**")
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏱️ {metadata.get('response_time', 0):.2f}s")
                with col2:
                    st.caption(f"📄 {metadata.get('retrieved_chunks', 0)} chunks")
                with col3:
                    st.caption(f"🤖 {st.session_state.selected_model}")
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                response_placeholder.markdown(error_msg)
                full_response = error_msg
            
            # Save message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "metadata": {
                    'sources': metadata.get('sources', []),
                    'retrieved_chunks': metadata.get('retrieved_chunks', 0),
                    'response_time': metadata.get('response_time', 0),
                    'model': st.session_state.selected_model
                }
            })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85rem;'>
    <b>🚀 Advanced RAG System</b><br>
    ChromaDB • Sentence Transformers • BM25 • Cross-Encoder Reranking • Groq API<br>
    <i>Hybrid Search • Semantic Chunking • Conversation Memory</i>
</div>
""", unsafe_allow_html=True)
