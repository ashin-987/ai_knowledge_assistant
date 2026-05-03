"""
Streamlit Web Interface - Cloud Version
Uses Hugging Face API instead of local Ollama
"""

import streamlit as st
import os
from pathlib import Path
import shutil

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine  # Using cloud version
import streamlit as st

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

# Title
st.title("🤖 AI Knowledge Assistant")
st.markdown("**Ask questions about your documents - 100% FREE & Cloud-Hosted!**")

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    
    # API token info
    has_token = bool(st.secrets.get("HUGGINGFACE_TOKEN"))
    if not has_token:
        st.warning("⚠️ Using public API (limited requests). Add HUGGINGFACE_TOKEN for unlimited access.")
        with st.expander("ℹ️ How to get a token (free)"):
            st.markdown("""
            1. Go to [huggingface.co](https://huggingface.co/join)
            2. Create free account
            3. Go to Settings → Access Tokens
            4. Create token and add to `.env` file:
               ```
               HUGGINGFACE_TOKEN=hf_your_token_here
               ```
            """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload your documents. The AI will read and understand them."
    )
    
    # Process button
    if st.button("🔄 Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing..."):
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
        else:
            st.warning("⚠️ Please upload files first")
    
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
    1. **Upload** your PDF/TXT files
    2. Click **Process Documents**
    3. **Ask questions** in the chat!
    
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
    st.info("👈 **Get Started:** Upload documents in the sidebar and click 'Process Documents'")
    
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
                if "loading" in result.get('error', '').lower():
                    st.warning(result['answer'])
                    st.info("⏳ The AI model is warming up. Wait 20 seconds and try again.")
                else:
                    st.error(f"⚠️ {result['answer']}")
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
