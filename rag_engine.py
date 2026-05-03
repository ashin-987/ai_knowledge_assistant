"""
RAG Engine - Cloud Version (Hugging Face API) - FIXED
This version fixes the 404 error and improves response parsing
"""

from typing import Dict
from vector_store import VectorStore
import requests
import os
import streamlit as st

class RAGEngine:
    def __init__(self, vector_store: VectorStore, model_name = "distilgpt2"):
        """
        Initialize the RAG engine with Hugging Face API.
        
        Args:
            vector_store: The vector database
            model_name: Hugging Face model to use (free tier)
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Get API token from Streamlit secrets (not .env!)
        try:
            self.api_token = st.secrets["HUGGINGFACE_TOKEN"]
        except:
            self.api_token = ""
        
        print(f"🤖 RAG Engine initialized with model: {model_name}")
        if not self.api_token:
            print("⚠️ Warning: No Hugging Face token found in Streamlit secrets!")
    
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
        
        # Step 3: Create prompt (improved format for better models)
        prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Instructions:
- If answer not found, say "I don't have enough information"
- Be concise and clear

Answer:
"""

        # Step 4: Call Hugging Face API
        print("🤖 Generating answer...")
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.3
                }
            }
            
            
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            print(f"API Response Status: {response.status_code}")

# Handle non-200
            if response.status_code != 200:
                return {
                    'answer': f"❌ API Error {response.status_code}:\n{response.text}",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': f"HTTP {response.status_code}"
                }

# Handle empty
            if not response.text.strip():
                return {
                    'answer': "⚠️ Empty response from Hugging Face API. Try again.",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': 'Empty response'
                }

# Parse JSON safely
            try:
                result = response.json()

                print("REACHED AFTER JSON PARSE")
                print("RAW RESPONSE:", response.text)
            except:
                return {
                    'answer': f"⚠️ Invalid response:\n{response.text}",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': 'Invalid JSON'
                }

# Extract answer
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', '')

            elif isinstance(result, dict):
                answer = result.get('generated_text') or result.get('error') or "No response"

            else:
                answer = str(result)

            answer = answer.replace('</s>', '').strip()

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
                'answer': f"❌ Error: {str(e)}",
                'sources': [],
                'retrieved_chunks': 0,
                'error': str(e)
            }
