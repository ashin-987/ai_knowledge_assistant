"""
Streamlit Web Interface - Cloud Version (FIXED)
Improved error handling and secrets management
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
    st.header("📁 Document Management")
    
    # API token status
    if has_token:
        st.success("✅ API Token: Configured")
    else:
        st.error("❌ API Token: Missing")
        with st.expander("🔧 How to configure token"):
            st.markdown("""
            **Streamlit Cloud Setup:**
            1. Go to your [Streamlit Cloud dashboard](https://share.streamlit.io/)
            2. Click on your app
            3. Settings → Secrets
            4. Add this:
            ```toml
            HUGGINGFACE_TOKEN = "hf_your_token_here"
            ```
            
            **Get a free token:**
            1. Create account at [huggingface.co](https://huggingface.co/join)
            2. Go to Settings → Access Tokens
            3. Create new token
            4. Copy and paste above
            """)
    
    st.divider()
    
    # File uploader
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
                        
                        # Initialize RAG engine
                        st.session_state.rag_engine = RAGEngine(
                            st.session_state.vector_store
                        )
                        st.session_state.documents_processed = True
                        
                        stats = st.session_state.vector_store.get_stats()
                        st.success(
                            f"✅ Processed {len(uploaded_files)} files "
                            f"({stats['total_chunks']} chunks)"
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
    2. **Upload** your PDF/TXT files
    3. Click **Process Documents**
    4. **Ask questions** in the chat!
    
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
                
                if "404" in error_type:
                    st.error("❌ **Model Not Found Error**")
                    st.markdown(result['answer'])
                    st.info("""
                    **Possible fixes:**
                    1. Update `rag_engine.py` to use a different model
                    2. Try: `HuggingFaceH4/zephyr-7b-beta`
                    3. Or: `mistralai/Mistral-7B-Instruct-v0.1`
                    """)
                    
                elif "loading" in error_type.lower():
                    st.warning(result['answer'])
                    st.info("⏳ The AI model is warming up. Wait 20 seconds and try again.")
                    
                elif "401" in error_type:
                    st.error("❌ **Authentication Error**")
                    st.markdown(result['answer'])
                    st.info("Check your Hugging Face token in Streamlit Cloud secrets")
                    
                else:
                    st.error(f"⚠️ Error: {result['answer']}")
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
