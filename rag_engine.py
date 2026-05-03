"""
RAG Engine - Cloud Version (Hugging Face API)
This version uses Hugging Face's free API instead of local Ollama
"""

from typing import Dict
from vector_store import VectorStore
import requests
import os

class RAGEngine:
    def __init__(self, vector_store: VectorStore, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the RAG engine with Hugging Face API.
        
        Args:
            vector_store: The vector database
            model_name: Hugging Face model to use (free tier)
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Get API token from environment variable
        self.api_token = os.environ.get("HUGGINGFACE_TOKEN", "")
        
        print(f"🤖 RAG Engine initialized with model: {model_name}")
        if not self.api_token:
            print("⚠️ Warning: No Hugging Face token found. Using public inference (may have rate limits)")
    
    def generate_answer(self, query: str, n_results: int = 3) -> Dict:
        """
        Generate an answer using RAG with Hugging Face API.
        
        Steps:
        1. Retrieve relevant chunks from vector database
        2. Create context from chunks
        3. Send to Hugging Face API
        4. Return answer with sources
        
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
        
        # Step 3: Create prompt
        prompt = f"""<s>[INST] You are a helpful AI assistant. Answer the question based ONLY on the context provided below.

Context from documents:
{context}

Question: {query}

Instructions:
- Only use information from the context above
- If the answer isn't in the context, say "I don't have enough information to answer that"
- Cite which documents you used (e.g., "According to Document 1...")
- Be concise and clear [/INST]"""

        # Step 4: Call Hugging Face API
        print("🤖 Generating answer...")
        try:
            headers = {}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True,
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract generated text
                if isinstance(result, list) and len(result) > 0:
                    answer = result[0].get('generated_text', '')
                    # Remove the prompt from the response
                    if '[/INST]' in answer:
                        answer = answer.split('[/INST]')[-1].strip()
                else:
                    answer = "Sorry, I couldn't generate a response."
                
                # Extract unique sources
                sources = list(set([doc['source'] for doc in retrieved_docs]))
                
                return {
                    'answer': answer,
                    'sources': sources,
                    'retrieved_chunks': len(retrieved_docs),
                    'context': context
                }
            
            elif response.status_code == 503:
                return {
                    'answer': "⏳ The AI model is loading. Please try again in 20 seconds.",
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': 'Model loading'
                }
            
            else:
                error_msg = f"API error: {response.status_code}"
                if response.status_code == 401:
                    error_msg = "❌ Invalid Hugging Face token. Please check your .env file."
                
                return {
                    'answer': error_msg,
                    'sources': [],
                    'retrieved_chunks': 0,
                    'error': error_msg
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
