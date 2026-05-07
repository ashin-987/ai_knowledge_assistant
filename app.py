"""
Streamlit Web Interface - Cloud Version (FIXED v2)
Added model selection and testing capabilities
"""

import streamlit as st
import os
from pathlib import Path
import shutil

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine

# Page configuration
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .error-box {
        padding: 1rem;
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .model-card {
        padding: 0.75rem;
        border-radius: 0.5rem;
        background-color: #f5f5f5;
        margin: 0.5rem 0;
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
    st.session_state.selected_model = "microsoft/Phi-3-mini-4k-instruct"  # Match rag_engine.py default

# Check for API token
def check_api_token():
    """Check if Hugging Face token is configured"""
    try:
        token = st.secrets["HUGGINGFACE_TOKEN"]
        return bool(token and token.startswith("hf_"))
    except:
        return False

has_token = check_api_token()

# Title
st.title("🤖 AI Knowledge Assistant")
st.markdown("**Ask questions about your documents - 100% FREE & Cloud-Hosted!**")

# Show token status prominently
if not has_token:
    st.error("⚠️ **CONFIGURATION REQUIRED:** Hugging Face token not found in Streamlit secrets!")
    st.info("""
    **To fix this:**
    1. Go to [Streamlit Cloud Dashboard](https://share.streamlit.io/)
    2. Open your app settings → Secrets
    3. Add: `HUGGINGFACE_TOKEN = "your_token_here"`
    4. Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    """)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API token status
    if has_token:
        st.success("✅ API Token: Configured")
    else:
        st.error("❌ API Token: Missing")
    
    st.divider()
    
    # Model Selection
    st.subheader("🤖 AI Model Selection")
    
    # Get models dynamically from RAGEngine
    model_options = {
        model_id: f"{info['name']}" 
        for model_id, info in RAGEngine.AVAILABLE_MODELS.items()
    }
    
    # Default to Flan-T5 Base (recommended in rag_engine.py)
    default_model = "microsoft/Phi-3-mini-4k-instruct"
    default_index = list(model_options.keys()).index(default_model) if default_model in model_options else 0
    
    selected_model = st.selectbox(
        "Choose AI Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=default_index,
        help="Choose the AI model for answering questions. Different models have different speeds and capabilities."
    )
    
    # Test Model button
    if st.button("🧪 Test Selected Model", use_container_width=True):
        with st.spinner(f"Testing {selected_model}..."):
            try:
                token = st.secrets.get("HUGGINGFACE_TOKEN", "")
            except:
                token = ""
            
            result = RAGEngine.test_model(selected_model, token)
            
            if result['status'] == 'success':
                st.success(f"✅ {result['message']}")
                st.session_state.selected_model = selected_model
            elif result['status'] == 'loading':
                st.warning(f"⏳ {result['message']}")
            else:
                st.error(f"❌ {result['message']}")
    
    # Show model info
    if selected_model in RAGEngine.AVAILABLE_MODELS:
        model_info = RAGEngine.AVAILABLE_MODELS[selected_model]
        with st.expander("ℹ️ Model Info"):
            st.markdown(f"**Name:** {model_info['name']}")
            st.markdown(f"**Description:** {model_info['description']}")
            st.markdown(f"**Max Tokens:** {model_info['max_tokens']}")
    
    st.divider()
    
    # File uploader
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload your documents. The AI will read and understand them."
    )
    
    # Process button
    if st.button("🔄 Process Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ Please upload files first")
        elif not has_token:
            st.error("❌ Cannot process without API token. Please configure Streamlit secrets first.")
        else:
            with st.spinner("Processing..."):
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
                    
                    # Process documents
                    processor = DocumentProcessor()
                    documents = processor.process_directory(upload_dir)
                    
                    if documents:
                        # Create vector store
                        st.session_state.vector_store = VectorStore()
                        st.session_state.vector_store.add_documents(documents)
                        
                        # Initialize RAG engine with selected model
                        st.session_state.rag_engine = RAGEngine(
                            st.session_state.vector_store,
                            model_name=selected_model
                        )
                        st.session_state.selected_model = selected_model
                        st.session_state.documents_processed = True
                        
                        stats = st.session_state.vector_store.get_stats()
                        model_name = RAGEngine.AVAILABLE_MODELS.get(selected_model, {}).get('name', selected_model)
                        st.success(
                            f"✅ Processed {len(uploaded_files)} files "
                            f"({stats['total_chunks']} chunks) with {model_name}"
                        )
                    else:
                        st.error("❌ No content found in files")
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.info("Check Streamlit logs for details")
    
    # Database stats
    if st.session_state.documents_processed:
        st.divider()
        st.subheader("📊 Database Stats")
        stats = st.session_state.vector_store.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", stats['total_chunks'])
        with col2:
            st.metric("Messages", len(st.session_state.messages))
        
        st.info(f"🤖 Using: {RAGEngine.AVAILABLE_MODELS.get(st.session_state.selected_model, {}).get('name', st.session_state.selected_model)}")
        
        # Clear buttons
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset DB", use_container_width=True):
                st.session_state.vector_store.reset()
                st.session_state.documents_processed = False
                st.session_state.messages = []
                st.rerun()
    
    # Instructions
    st.divider()
    st.subheader("ℹ️ How to Use")
    st.markdown("""
    1. **Configure** API token in Streamlit secrets
    2. **Test** your selected AI model
    3. **Upload** your PDF/TXT files
    4. Click **Process Documents**
    5. **Ask questions** in the chat!
    
    **Example Questions:**
    - "Summarize the main points"
    - "What does it say about X?"
    - "Find information on Y"
    - "Compare A and B"
    """)
    
    # Footer
    st.divider()
    st.caption("💡 100% Free & Cloud-Hosted")
    st.caption("Powered by Hugging Face")

# Main chat area
if not st.session_state.documents_processed:
    # Welcome message
    if has_token:
        st.info("👈 **Get Started:** Upload documents in the sidebar and click 'Process Documents'")
    else:
        st.warning("⚠️ **Setup Required:** Configure your Hugging Face token in Streamlit Cloud secrets (see sidebar)")
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🆓 Free")
        st.markdown("No costs, hosted on cloud")
    
    with col2:
        st.markdown("### 🌐 Online")
        st.markdown("Share the link with anyone")
    
    with col3:
        st.markdown("### 🚀 Fast")
        st.markdown("Semantic search with RAG")
    
    # Model comparison
    st.divider()
    st.subheader("🤖 Available Models")
    
    # Display models dynamically from RAGEngine
    for model_id, model_info in RAGEngine.AVAILABLE_MODELS.items():
        with st.expander(f"**{model_info['name']}**" + (" *(Recommended)*" if "RECOMMENDED" in model_info['description'] else "")):
            st.markdown(f"**Description:** {model_info['description']}")
            st.markdown(f"**Max Tokens:** {model_info['max_tokens']}")
            st.markdown(f"**Model ID:** `{model_id}`")
    
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources
            if "sources" in message and message["sources"]:
                with st.expander("📎 View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- 📄 {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.rag_engine.generate_answer(prompt)
            
            # Check for errors
            if 'error' in result:
                error_type = result.get('error', '')
                
                if "Model Not Found" in error_type:
                    st.error("❌ **Model Not Available**")
                    st.markdown(result['answer'])
                    st.info("💡 Try selecting a different model in the sidebar")
                    
                elif "Model Loading" in error_type:
                    st.warning(result['answer'])
                    st.info("⏳ Wait 20-30 seconds, then try your question again.")
                    
                elif "Authentication" in error_type:
                    st.error("❌ **Authentication Error**")
                    st.markdown(result['answer'])
                    st.info("Check your Hugging Face token in Streamlit Cloud secrets")
                    
                else:
                    st.error(f"⚠️ {result['answer']}")
                    with st.expander("🔍 Debug Info"):
                        st.code(result.get('error', 'Unknown error'))
            else:
                # Display answer
                st.markdown(result['answer'])
                
                # Show sources
                if result['sources']:
                    with st.expander("📎 View Sources"):
                        st.markdown(f"*Used {result['retrieved_chunks']} chunks from:*")
                        for source in result['sources']:
                            st.markdown(f"- 📄 {source}")
                
                # Save message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": result.get('sources', [])
                })

# Bottom info
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    Built with Python • ChromaDB • Sentence Transformers • Hugging Face API<br>
    <b>100% FREE • CLOUD-HOSTED • SHAREABLE</b>
</div>
""", unsafe_allow_html=True)