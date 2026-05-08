"""
RAG Engine - Cloud Version with FIXED Hugging Face API Integration
Uses the correct serverless inference API format
"""

from typing import Dict
from vector_store import VectorStore
import os
import streamlit as st
from groq import Groq

class RAGEngine:
    # Working models VERIFIED with serverless inference API (tested and working!)
    AVAILABLE_MODELS = {
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B Instant",
            "description": "Fast Groq model",
            "max_tokens": 512
        },

        "llama3-70b-8192": {
            "name": "Llama 3 70B",
            "description": "More powerful Groq model",
            "max_tokens": 1024
        }
    }
    
    def __init__(self, vector_store: VectorStore, model_name="llama-3.1-8b-instant"):

        self.vector_store = vector_store
        self.model_name = model_name

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

        # Initialize Groq client
        if not self.api_key:
            raise ValueError("Groq API key missing")

        self.client = Groq(api_key=self.api_key)

        print(f"🤖 RAG Engine initialized with: {model_name}")
    
    
    
    def generate_answer(self, query: str, n_results: int = 5) -> Dict:
        """
        Generate an answer using RAG with Groq API.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve (increased from 2 to 5)
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        
        print(f"\n🔍 Processing query: '{query}'")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query, n_results=n_results)
        
        if not retrieved_docs:
            return {
                'answer': "❌ I couldn't find any relevant information in your documents.",
                'sources': [],
                'retrieved_chunks': 0
            }
        
        print(f"📚 Retrieved {len(retrieved_docs)} relevant chunks")
        
        # Step 2: Build context
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Source {i}]: {doc['text'][:300]}")  # Limit context length
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Create prompt based on model type
        prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        # Step 4: Call API with retries
        # Step 4: Generate answer with Groq
        try:

            print("🚀 Calling Groq API...")

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=512
            )

            answer = completion.choices[0].message.content

            sources = list(set([doc['source'] for doc in retrieved_docs]))

            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": len(retrieved_docs)
            }

        except Exception as e:
            print(f"Groq Error: {e}")

            return {
                "answer": f"❌ Groq API Error: {str(e)}",
                "sources": [],
                "retrieved_chunks": 0,
                "error": str(e)
            }
