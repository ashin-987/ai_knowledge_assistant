"""
Vector Store - Handles embeddings and vector database
Author: Your Name
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import hashlib

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        """
        Initialize the vector database.
        
        Args:
            persist_directory: Where to save the database
        """
        print("🔧 Initializing Vector Store...")
        
        # Create/open ChromaDB database
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Load embedding model (FREE - runs locally)
        print("📦 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"description": "Document embeddings"}
        )
        print("📊 Existing chunks:", self.collection.count())
        print("✅ Vector Store ready!")
    
    def add_documents(self, documents: List[Dict]):
        """Add document chunks to the vector database."""
        if not documents:
            print("⚠️ No documents to add!")
            return
        
        print(f"📝 Adding {len(documents)} chunks to database...")
        
        # Extract data
        texts = [doc['text'] for doc in documents]
        metadatas = [
            {'source': doc['source'], 'chunk_id': doc['chunk_id']} 
            for doc in documents
        ]
        
        # Generate unique IDs
        ids = [
            hashlib.md5(f"{doc['source']}_{doc['chunk_id']}".encode()).hexdigest()
            for doc in documents
        ]
        
        # Create embeddings
        print("🔮 Creating embeddings (this may take a minute)...")
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True
        ).tolist()
        
        # Add to database
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Successfully added {len(documents)} chunks!")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for relevant document chunks.
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        # Convert query to embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        # Format results
        formatted_results = []

        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]

                # Optional: filter out extremely weak matches (distance > 1.5 is very dissimilar)
                if distance > 1.5:
                    continue

                formatted_results.append({
                    'text': results['documents'][0][i],
                    'source': results['metadatas'][0][i]['source'],
                    'distance': distance
                })

        return formatted_results
    
    def get_stats(self):
        """Get database statistics."""
        return {
            'total_chunks': self.collection.count(),
            'collection_name': self.collection.name
        }
    
    def reset(self):
        """Delete all data and start fresh."""
        self.client.delete_collection("knowledge_base")
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"description": "Document embeddings"}
        )
        print("🗑️ Database reset!")
