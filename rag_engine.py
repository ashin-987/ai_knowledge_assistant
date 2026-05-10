"""
Improved RAG Engine - Advanced Features
UPDATED: May 2026 - Current Active Groq Models
Includes: Streaming, conversation memory, query expansion, better prompts
"""

from typing import Dict, List, Optional, Generator
from vector_store import VectorStore
import os
import streamlit as st
from groq import Groq
from datetime import datetime
import time

class RAGEngine:
    # ===================================================================
    # UPDATED MAY 2026: Currently Active Groq Models
    # ===================================================================
    # All old models (mixtral, llama3-70b, llama3-8b, llama-3.1-70b) 
    # have been DECOMMISSIONED
    # ===================================================================
    
    AVAILABLE_MODELS = {
        # ============ PRODUCTION MODELS (Stable & Recommended) ============
        
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B Instant ⚡",
            "description": "⚡ Fastest model - 560 tokens/sec - Best for real-time chat",
            "max_tokens": 131072,
            "speed": "Very Fast",
            "status": "✅ Production"
        },
        
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B Versatile 🎯",
            "description": "🎯 Best quality open-source - 280 tokens/sec - Recommended for most uses",
            "max_tokens": 131072,
            "speed": "Fast",
            "status": "✅ Production"
        },
        
        "openai/gpt-oss-120b": {
            "name": "GPT-OSS 120B 🧠",
            "description": "🧠 Highest intelligence - 500 tokens/sec - Advanced reasoning",
            "max_tokens": 131072,
            "speed": "Fast",
            "status": "✅ Production"
        },
        
        "openai/gpt-oss-20b": {
            "name": "GPT-OSS 20B 🚀",
            "description": "🚀 Ultra-fast - 1000 tokens/sec - Great balance",
            "max_tokens": 131072,
            "speed": "Blazing Fast",
            "status": "✅ Production"
        }
    }
    
    def __init__(self, vector_store: VectorStore, model_name="llama-3.1-8b-instant"):
        """
        Initialize the improved RAG engine.
        
        Args:
            vector_store: ImprovedVectorStore instance
            model_name: Groq model to use (default: llama-3.1-8b-instant)
        """
        self.vector_store = vector_store
        self.model_name = model_name
        
        # Conversation memory
        self.conversation_history = []
        self.max_history = 5  # Keep last 5 exchanges
        
        # Get API key
        try:
            self.api_key = st.secrets["GROQ_API_KEY"]
            print("✅ Using Streamlit secrets Groq key")
        except Exception:
            self.api_key = os.getenv("GROQ_API_KEY", "")
            if self.api_key:
                print("✅ Using .env Groq key")
            else:
                print("⚠️ No Groq API key found")

        if not self.api_key:
            raise ValueError("Groq API key missing! Set GROQ_API_KEY in .env or Streamlit secrets")

        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        
        # Validate model
        if model_name not in self.AVAILABLE_MODELS:
            print(f"⚠️ Warning: Model {model_name} not in known models list")
            print(f"📝 Available models: {', '.join(self.AVAILABLE_MODELS.keys())}")
        
        print(f"🤖 Improved RAG Engine initialized with: {model_name}")
        if model_name in self.AVAILABLE_MODELS:
            status = self.AVAILABLE_MODELS[model_name]['status']
            if status == "⚠️ Preview":
                print(f"⚠️ WARNING: {model_name} is in PREVIEW - may be deprecated soon!")
    
    def add_to_history(self, user_query: str, assistant_response: str):
        """Add exchange to conversation history."""
        self.conversation_history.append({
            'user': user_query,
            'assistant': assistant_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_context(self) -> str:
        """Build conversation context from history."""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(context_parts)
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate query variations for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations including original
        """
        try:
            expansion_prompt = f"""Generate 2 alternative ways to ask this question. Keep them concise.

Original question: {query}

Alternative 1:
Alternative 2:"""
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Use fast model for expansion
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            alternatives_text = response.choices[0].message.content
            
            # Parse alternatives
            alternatives = [query]  # Start with original
            for line in alternatives_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('Alternative'):
                    # Remove numbering and clean up
                    cleaned = line.lstrip('12.:- ')
                    if cleaned and len(cleaned) > 10:
                        alternatives.append(cleaned)
            
            return alternatives[:3]  # Return max 3 variations
            
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            return [query]  # Fall back to original
    
    def build_context(self, retrieved_docs: List[Dict], max_context_length: int = 4000) -> str:
        """
        Build context from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents
            max_context_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Format document with source info
            doc_text = f"[Source {i}: {doc['source']}]\n{doc['text']}\n"
            doc_length = len(doc_text)
            
            # Check if adding this would exceed limit
            if current_length + doc_length > max_context_length:
                # Truncate this document to fit
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if meaningful space left
                    doc_text = doc_text[:remaining] + "...\n"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += doc_length
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str, conversation_context: str = "") -> str:
        """
        Create an optimized prompt for the LLM.
        
        Args:
            query: User's question
            context: Retrieved document context
            conversation_context: Previous conversation
            
        Returns:
            Formatted prompt
        """
        # Base system instruction
        system_instruction = """You are a helpful AI assistant that answers questions based on provided documents.

IMPORTANT RULES:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain the answer, say so clearly
3. Be concise but complete
4. Cite sources when possible (e.g., "According to [Source 1]...")
5. If the question is unclear, ask for clarification"""

        # Build prompt with conversation context if available
        if conversation_context:
            prompt = f"""{system_instruction}

PREVIOUS CONVERSATION:
{conversation_context}

DOCUMENT CONTEXT:
{context}

CURRENT QUESTION: {query}

ANSWER:"""
        else:
            prompt = f"""{system_instruction}

DOCUMENT CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def generate_answer(self, query: str, n_results: int = 5, 
                       use_rerank: bool = True, 
                       use_query_expansion: bool = False) -> Dict:
        """
        Generate an answer using advanced RAG techniques.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            use_rerank: Whether to use reranking
            use_query_expansion: Whether to use query expansion
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        print(f"\n🔍 Processing query: '{query}'")
        
        # Step 1: Query expansion (optional)
        queries = [query]
        if use_query_expansion:
            print("🔄 Expanding query...")
            queries = self.expand_query(query)
            print(f"📝 Generated {len(queries)} query variations")
        
        # Step 2: Retrieve documents for all query variations
        all_retrieved = []
        for q in queries:
            if use_rerank:
                # Use full pipeline: hybrid search + rerank
                docs = self.vector_store.search_with_rerank(
                    q, 
                    n_results=n_results,
                    initial_results=n_results * 4
                )
            else:
                # Use hybrid search only
                docs = self.vector_store.hybrid_search(q, n_results=n_results)
            
            all_retrieved.extend(docs)
        
        # Deduplicate by text content
        seen_texts = set()
        unique_docs = []
        for doc in all_retrieved:
            if doc['text'] not in seen_texts:
                seen_texts.add(doc['text'])
                unique_docs.append(doc)
        
        # Take top N after deduplication
        retrieved_docs = unique_docs[:n_results]
        
        if not retrieved_docs:
            return {
                'answer': "❌ I couldn't find any relevant information in the documents to answer your question.",
                'sources': [],
                'retrieved_chunks': 0,
                'response_time': time.time() - start_time
            }
        
        print(f"📚 Retrieved {len(retrieved_docs)} unique chunks")
        
        # Step 3: Build context
        context = self.build_context(retrieved_docs)
        conversation_context = self.get_conversation_context()
        
        # Step 4: Create prompt
        prompt = self.create_prompt(query, context, conversation_context)
        
        # Step 5: Generate answer
        try:
            print(f"🚀 Calling Groq API with model: {self.model_name}")
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024
            )
            
            answer = completion.choices[0].message.content
            
            # Extract unique sources
            sources = list(set([doc['source'] for doc in retrieved_docs]))
            
            # Add to conversation history
            self.add_to_history(query, answer)
            
            response_time = time.time() - start_time
            
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": len(retrieved_docs),
                "response_time": response_time,
                "model": self.model_name,
                "retrieval_scores": [
                    {
                        'source': doc['source'],
                        'score': doc.get('rerank_score', doc.get('combined_score', 0))
                    }
                    for doc in retrieved_docs[:3]  # Top 3
                ]
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Groq Error: {error_msg}")
            
            # Check for specific errors
            if "400" in error_msg and "decommissioned" in error_msg.lower():
                return {
                    "answer": f"❌ Model Error: The model '{self.model_name}' has been decommissioned by Groq.\n\nPlease select a different model from the sidebar. Recommended: llama-3.1-8b-instant or llama-3.3-70b-versatile",
                    "sources": [],
                    "retrieved_chunks": 0,
                    "response_time": time.time() - start_time,
                    "error": "Model decommissioned"
                }
            
            return {
                "answer": f"❌ Error generating response: {error_msg}\n\nTry selecting a different model from the sidebar.",
                "sources": [],
                "retrieved_chunks": 0,
                "response_time": time.time() - start_time,
                "error": error_msg
            }
    
    def generate_answer_streaming(self, query: str, n_results: int = 5) -> Generator[str, None, Dict]:
        """
        Generate streaming answer for real-time display.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            
        Yields:
            Text chunks as they're generated
            
        Returns:
            Final metadata dictionary
        """
        start_time = time.time()
        
        # Retrieve documents
        retrieved_docs = self.vector_store.search_with_rerank(
            query,
            n_results=n_results,
            initial_results=n_results * 4
        )
        
        if not retrieved_docs:
            yield "❌ I couldn't find any relevant information in the documents."
            return {
                'sources': [],
                'retrieved_chunks': 0,
                'response_time': time.time() - start_time
            }
        
        # Build context and prompt
        context = self.build_context(retrieved_docs)
        conversation_context = self.get_conversation_context()
        prompt = self.create_prompt(query, context, conversation_context)
        
        # Generate streaming response
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                stream=True
            )
            
            full_answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield content
            
            # Add to history
            self.add_to_history(query, full_answer)
            
            # Return metadata
            sources = list(set([doc['source'] for doc in retrieved_docs]))
            return {
                'sources': sources,
                'retrieved_chunks': len(retrieved_docs),
                'response_time': time.time() - start_time,
                'full_answer': full_answer
            }
            
        except Exception as e:
            error_msg = str(e)
            
            if "400" in error_msg and "decommissioned" in error_msg.lower():
                yield f"\n\n❌ Model Error: '{self.model_name}' has been decommissioned.\nPlease select a different model from the sidebar."
            else:
                yield f"\n\n❌ Error: {error_msg}"
            
            return {
                'sources': [],
                'retrieved_chunks': 0,
                'response_time': time.time() - start_time,
                'error': error_msg
            }
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared")
    
    def get_suggested_questions(self, n_questions: int = 3) -> List[str]:
        """
        Generate suggested questions based on document content.
        
        Args:
            n_questions: Number of questions to generate
            
        Returns:
            List of suggested questions
        """
        # Get a sample of documents
        sample_docs = self.vector_store.search_semantic("summary overview", n_results=3)
        
        if not sample_docs:
            return []
        
        # Build sample context
        sample_text = " ".join([doc['text'][:200] for doc in sample_docs])
        
        try:
            prompt = f"""Based on this document content, suggest {n_questions} interesting questions someone might ask.
Make them specific and relevant.

Content: {sample_text}

Questions (one per line):"""
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            questions_text = response.choices[0].message.content
            
            # Parse questions
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if line and '?' in line:
                    # Remove numbering
                    cleaned = line.lstrip('0123456789.:- ')
                    if cleaned:
                        questions.append(cleaned)
            
            return questions[:n_questions]
            
        except Exception as e:
            print(f"⚠️ Failed to generate suggestions: {e}")
            return []
