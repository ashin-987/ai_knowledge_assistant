"""
RAG Engine - Cloud Version (Hugging Face API) - FIXED v2
Updated with working models and improved error handling
"""

from typing import Dict
from vector_store import VectorStore
import requests
import os
import streamlit as st

class RAGEngine:
    # List of working models (tested and verified)
    AVAILABLE_MODELS = {
        "google/flan-t5-base": {
            "name": "Google Flan-T5 Base",
            "description": "Fast, reliable, good for Q&A",
            "max_tokens": 512
        },
        "google/flan-t5-large": {
            "name": "Google Flan-T5 Large",
            "description": "Better quality than base, still fast",
            "max_tokens": 512
        }
    }
    
    def __init__(self, vector_store: VectorStore, model_name="google/flan-t5-base"):
        """
        Initialize the RAG engine with Hugging Face API.
        
        Args:
            vector_store: The vector database
            model_name: Hugging Face model to use
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        print("API URL BEING USED:", self.api_url)
        
        # Get API token from Streamlit secrets
        try:
            self.api_token = st.secrets["HUGGINGFACE_TOKEN"]
        except:
            self.api_token = ""
        
        print(f"🤖 RAG Engine initialized with model: {model_name}")
        if not self.api_token:
            print("⚠️ Warning: No Hugging Face token found in Streamlit secrets!")
    
    @staticmethod
    def test_model(model_name: str, api_token: str = "") -> Dict:
        """
        Test if a model is available and responding.
        
        Args:
            model_name: Model identifier
            api_token: Optional API token
            
        Returns:
            Dict with status and message
        """
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        
        payload = {"inputs": "Hello, how are you?"}
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                return {"status": "success", "message": "Model is working!"}
            elif response.status_code == 503:
                return {"status": "loading", "message": "Model is loading, try again in 20 seconds"}
            elif response.status_code == 404:
                return {"status": "error", "message": "Model not found or not available"}
            elif response.status_code == 401:
                return {"status": "error", "message": "Authentication failed - check your token"}
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
                
        except requests.exceptions.Timeout:
            return {"status": "timeout", "message": "Request timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def generate_answer(self, query: str, n_results: int = 3) -> Dict:
        """
        Generate an answer using RAG with Hugging Face API.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        
        print(f"\n🔍 Searching for: '{query}'")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query, n_results=n_results)
        
        if not retrieved_docs:
            return {
                'answer': "❌ I couldn't find any relevant information in your documents.",
                'sources': [],
                'retrieved_chunks': 0
            }
        
        print(f"📚 Found {len(retrieved_docs)} relevant chunks")
        
        # Step 2: Build context
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Document {i}: {doc['source']}]\n{doc['text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Create prompt - optimized for different models
        if "flan-t5" in self.model_name.lower():
            # T5 models prefer simpler prompts
            prompt = f"""Answer the question based on the context below.

Context: {context}

Question: {query}

Answer:"""
        else:
            # For instruction-following models
            prompt = f"""You are a helpful AI assistant. Answer the question using ONLY the context provided.

Context:
{context}

Question: {query}

Instructions:
- Answer based only on the context
- Be concise and clear
- If you can't find the answer, say "I don't have enough information"

Answer:"""

        # Step 4: Call Hugging Face API
        print("🤖 Generating answer...")
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            
            # Adjust parameters based on model
            max_length = self.AVAILABLE_MODELS.get(self.model_name, {}).get("max_tokens", 512)
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )


            print("STATUS:", response.status_code)
            print("RAW RESPONSE:", response.text[:500])

# Try parsing JSON safely FIRST
            try:
                result = response.json()
            except Exception:
                return {
                    "answer": f"⚠️ Non-JSON response from API:\n{response.text[:300]}",
                    "sources": [],
                    "retrieved_chunks": 0,
                    "error": f"Invalid JSON ({response.status_code})"
                }

# Handle loading
            if response.status_code == 503:
                return {
                    "answer": "⏳ Model is loading. Please wait 20–30 seconds and try again.",
                    "sources": [],
                    "retrieved_chunks": 0,
                    "error": "Model loading"
                }

# Handle auth error
            if response.status_code == 401:
                return {
                    "answer": "❌ Invalid Hugging Face token. Check Streamlit secrets.",
                    "sources": [],
                    "retrieved_chunks": 0,
                    "error": "Authentication failed"
                }

# SUCCESS
            if response.status_code == 200:
                if isinstance(result, list) and len(result) > 0:
                    answer = result[0].get("generated_text", "")
                else:
                    answer = str(result)

                sources = list(set([doc['source'] for doc in retrieved_docs]))

                return {
                    "answer": answer.strip(),
                    "sources": sources,
                    "retrieved_chunks": len(retrieved_docs)
                }

# EVERYTHING ELSE (DON'T assume model error)
            return {
                "answer": f"❌ API Error {response.status_code}:\n{response.text[:300]}",
                "sources": [],
                "retrieved_chunks": 0,
                "error": response.text
            }
                
        except requests.exceptions.Timeout:
            return {
                'answer': "⏰ Request timed out. The model might be busy. Please try again.",
                'sources': [],
                'retrieved_chunks': 0,
                'error': 'Timeout'
            }
        
        except Exception as e:
            return {
                'answer': f"❌ Unexpected error: {str(e)}",
                'sources': [],
                'retrieved_chunks': 0,
                'error': str(e)
            }
