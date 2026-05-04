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
        "mistralai/Mistral-7B-Instruct-v0.1": {
            "name": "Mistral 7B Instruct",
            "description": "High quality, may take time to load",
            "max_tokens": 1024
        },
        "google/flan-t5-large": {
            "name": "Google Flan-T5 Large",
            "description": "Better quality than base, still fast",
            "max_tokens": 512
        },
        "HuggingFaceH4/zephyr-7b-beta": {
            "name": "Zephyr 7B Beta",
            "description": "High quality but may not always be available",
            "max_tokens": 1024
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
                timeout=30
            )

            print(f"API Response Status: {response.status_code}")

            # Handle different response codes
            if response.status_code == 503:
                return {
                    'answer': "⏳ The AI model is currently loading. Please wait 20-30 seconds and try again.",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': 'Model Loading'
                }
            
            if response.status_code == 404:
                return {
                    'answer': f"❌ Model '{self.model_name}' not found or unavailable.\n\nPlease select a different model in the sidebar.",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': 'Model Not Found'
                }
            
            if response.status_code == 401:
                return {
                    'answer': "❌ Authentication failed. Please check your Hugging Face token in Streamlit secrets.",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': 'Authentication Error'
                }
            
            if response.status_code != 200:
                return {
                    'answer': f"❌ API Error {response.status_code}:\n{response.text[:300]}",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': f"HTTP {response.status_code}"
                }

            # Check for empty response
            if not response.text.strip():
                return {
                    'answer': "⚠️ Empty response from API. Try again.",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': 'Empty response'
                }

            # Parse JSON response
            try:
                result = response.json()
            except:
                return {
                    'answer': f"⚠️ Invalid JSON response:\n{response.text[:300]}",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': 'Invalid JSON'
                }

            # Extract answer based on response structure
            answer = ""
            
            if isinstance(result, list) and len(result) > 0:
                # Most models return a list
                if isinstance(result[0], dict):
                    answer = result[0].get('generated_text', '')
                else:
                    answer = str(result[0])
                    
            elif isinstance(result, dict):
                # Some models return a dict
                answer = result.get('generated_text') or result.get('error') or str(result)
            else:
                answer = str(result)

            # Clean up answer
            answer = answer.replace('</s>', '').replace('<pad>', '').strip()
            
            # Remove the prompt from the answer if model repeated it
            if prompt in answer:
                answer = answer.replace(prompt, '').strip()

            # Extract sources
            sources = list(set([doc['source'] for doc in retrieved_docs]))

            return {
                'answer': answer if answer else "I don't have enough information to answer that.",
                'sources': sources,
                'retrieved_chunks': len(retrieved_docs),
                'context': context
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
