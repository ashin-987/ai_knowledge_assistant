"""
RAG Engine - Cloud Version with FIXED Hugging Face API Integration
Uses the correct serverless inference API format
"""

from typing import Dict
from vector_store import VectorStore
import requests
import os
import streamlit as st
import time
import json

class RAGEngine:
    # Working models verified with serverless inference API
    AVAILABLE_MODELS = {
        "meta-llama/Llama-3.2-3B-Instruct": {
            "name": "Llama 3.2 3B Instruct",
            "description": "Fast, instruction-following model - RECOMMENDED",
            "max_tokens": 512,
            "api_type": "text_generation"
        },
        "microsoft/Phi-3.5-mini-instruct": {
            "name": "Phi-3.5 Mini Instruct",
            "description": "Compact, efficient, good quality",
            "max_tokens": 512,
            "api_type": "text_generation"
        },
        "HuggingFaceH4/zephyr-7b-beta": {
            "name": "Zephyr 7B Beta",
            "description": "Powerful open-source model",
            "max_tokens": 512,
            "api_type": "text_generation"
        },
        "google/flan-t5-large": {
            "name": "Google Flan-T5 Large",
            "description": "Reliable text generation",
            "max_tokens": 512,
            "api_type": "text_generation"
        }
    }
    
    def __init__(self, vector_store: VectorStore, model_name="microsoft/Phi-3.5-mini-instruct"):
        """
        Initialize the RAG engine with Hugging Face Serverless Inference API.
        
        Args:
            vector_store: The vector database
            model_name: Hugging Face model to use
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.model_info = self.AVAILABLE_MODELS.get(model_name, {})
        self.api_type = self.model_info.get("api_type", "text_generation")
        
        # Use the correct serverless inference endpoint
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        print(f"🔗 API URL: {self.api_url}")
        
        # Get API token from Streamlit secrets
        try:
            self.api_token = st.secrets["HUGGINGFACE_TOKEN"]
            if self.api_token and len(self.api_token) > 10:
                print("✅ API Token found")
            else:
                print("⚠️ API Token looks invalid")
                self.api_token = ""
        except Exception as e:
            print(f"⚠️ Could not get token: {e}")
            self.api_token = ""
        
        print(f"🤖 RAG Engine initialized with: {model_name}")
    
    @staticmethod
    def test_model(model_name: str, api_token: str = "") -> Dict:
        """
        Test if a model is available and responding.
        
        Args:
            model_name: Model identifier
            api_token: API token
            
        Returns:
            Dict with status and message
        """
        model_info = RAGEngine.AVAILABLE_MODELS.get(model_name, {})
        api_type = model_info.get("api_type", "text_generation")
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        headers = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        
        # Use appropriate payload based on model type
        if api_type == "messages":
            payload = {
                "inputs": "Hello, how are you?",
                "parameters": {"max_new_tokens": 50}
            }
        else:
            payload = {
                "inputs": "Hello, how are you?",
                "parameters": {"max_length": 50}
            }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=15)
            
            print(f"Test response status: {response.status_code}")
            print(f"Test response: {response.text[:200]}")
            
            if response.status_code == 200:
                # Try to parse response
                try:
                    result = response.json()
                    return {"status": "success", "message": "Model is working!", "response": result}
                except:
                    return {"status": "success", "message": "Model responded (non-JSON)", "response": response.text[:100]}
            elif response.status_code == 503:
                return {"status": "loading", "message": "Model is loading, try again in 20-30 seconds"}
            elif response.status_code == 404:
                return {"status": "error", "message": f"Model not found at {api_url}"}
            elif response.status_code == 401 or response.status_code == 403:
                return {"status": "error", "message": "Authentication failed - check your token"}
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
                
        except requests.exceptions.Timeout:
            return {"status": "timeout", "message": "Request timed out - model might be busy"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}
    
    def generate_answer(self, query: str, n_results: int = 3) -> Dict:
        """
        Generate an answer using RAG with Hugging Face API.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            
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
        if self.api_type == "messages":
            # For instruction-following models (Llama, Phi, Zephyr)
            prompt = f"""You are a helpful AI assistant. Answer the question using ONLY the information provided in the context below.

Context:
{context}

Question: {query}

Provide a clear, concise answer based solely on the context. If the context doesn't contain enough information, say "I don't have enough information to answer that."

Answer:"""
        else:
            # For text generation models (T5, BART)
            prompt = f"""Context: {context}

Question: {query}

Answer based on the context:"""

        # Step 4: Call API with retries
        max_retries = 2
        for attempt in range(max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_token:
                    headers["Authorization"] = f"Bearer {self.api_token}"
                
                max_tokens = self.model_info.get("max_tokens", 512)
                
                # Prepare payload
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
                
                print(f"🚀 Calling API (attempt {attempt + 1}/{max_retries})...")
                print(f"📍 URL: {self.api_url}")
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                print(f"📊 Status: {response.status_code}")
                print(f"📝 Response preview: {response.text[:200]}")
                
                # Handle model loading
                if response.status_code == 503:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        pass
                    
                    if "loading" in str(error_data).lower() or "warming up" in response.text.lower():
                        if attempt < max_retries - 1:
                            print(f"⏳ Model loading, waiting 20 seconds...")
                            time.sleep(20)
                            continue
                        else:
                            return {
                                "answer": "⏳ Model is still loading. Please wait 30 seconds and try again.",
                                "sources": [],
                                "retrieved_chunks": 0,
                                "error": "Model loading"
                            }
                
                # Handle authentication errors
                if response.status_code in [401, 403]:
                    return {
                        "answer": "❌ Authentication failed. Please check your Hugging Face token in Streamlit secrets.",
                        "sources": [],
                        "retrieved_chunks": 0,
                        "error": "Authentication failed"
                    }
                
                # Handle 404 errors
                if response.status_code == 404:
                    return {
                        "answer": f"❌ Model not found or not available via Serverless API.\n\nTried: {self.model_name}\n\nTry selecting a different model from the sidebar.",
                        "sources": [],
                        "retrieved_chunks": 0,
                        "error": f"404 - Model not found at {self.api_url}"
                    }
                
                # Success - parse response
                if response.status_code == 200:
                    try:
                        result = response.json()
                        
                        # Extract answer based on response format
                        answer = ""
                        if isinstance(result, list) and len(result) > 0:
                            if isinstance(result[0], dict):
                                answer = result[0].get("generated_text", "")
                            else:
                                answer = str(result[0])
                        elif isinstance(result, dict):
                            answer = result.get("generated_text", str(result))
                        else:
                            answer = str(result)
                        
                        if not answer or answer.strip() == "":
                            answer = "⚠️ Model returned empty response. Try asking the question differently."
                        
                        sources = list(set([doc['source'] for doc in retrieved_docs]))
                        
                        return {
                            "answer": answer.strip(),
                            "sources": sources,
                            "retrieved_chunks": len(retrieved_docs)
                        }
                    
                    except json.JSONDecodeError:
                        return {
                            "answer": f"⚠️ Received non-JSON response:\n{response.text[:300]}",
                            "sources": [],
                            "retrieved_chunks": 0,
                            "error": "Invalid JSON response"
                        }
                
                # Other errors
                return {
                    "answer": f"❌ API Error {response.status_code}:\n{response.text[:300]}",
                    "sources": [],
                    "retrieved_chunks": 0,
                    "error": f"HTTP {response.status_code}"
                }
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"⏰ Timeout, retrying...")
                    time.sleep(5)
                    continue
                else:
                    return {
                        'answer': "⏰ Request timed out after multiple attempts. The model might be busy.",
                        'sources': [],
                        'retrieved_chunks': 0,
                        'error': 'Timeout'
                    }
            
            except Exception as e:
                print(f"❌ Exception: {e}")
                return {
                    'answer': f"❌ Unexpected error: {str(e)}",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': str(e)
                }
        
        return {
            'answer': "❌ Failed after multiple attempts",
            'sources': [],
            'retrieved_chunks': 0,
            'error': 'Max retries exceeded'
        }