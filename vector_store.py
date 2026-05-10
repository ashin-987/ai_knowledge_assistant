"""
Improved Vector Store - Hybrid Search with BM25 + Semantic Search
Includes: Reranking, persistence fix, and advanced retrieval
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Optional
import hashlib
import os
import numpy as np
from rank_bm25 import BM25Okapi
import json

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        """
        Initialize the vector database with hybrid search capabilities.
        
        Args:
            persist_directory: Where to save the database
        """
        print("🔧 Initializing Improved Vector Store...")
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist (DON'T DELETE IT!)
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create/open ChromaDB database
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Load embedding model (FREE - runs locally)
        print("📦 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load reranker model for precision
        print("🎯 Loading reranker model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"description": "Document embeddings with hybrid search"}
        )
        
        # BM25 index (in-memory, rebuilt from DB if exists)
        self.bm25_index = None
        self.documents_list = []
        self.document_metadata = []
        
        # Load existing documents for BM25
        self._rebuild_bm25_from_db()
        
        print(f"📊 Existing chunks: {self.collection.count()}")
        print("✅ Improved Vector Store ready with hybrid search!")
    
    def _rebuild_bm25_from_db(self):
        """Rebuild BM25 index from existing ChromaDB data"""
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                print(f"🔄 Rebuilding BM25 index from {existing_count} existing documents...")
                
                # Get all documents from ChromaDB
                results = self.collection.get(
                    include=['documents', 'metadatas']
                )
                
                if results['documents']:
                    self.documents_list = results['documents']
                    self.document_metadata = results['metadatas']
                    
                    # Build BM25 index
                    tokenized_corpus = [doc.lower().split() for doc in self.documents_list]
                    self.bm25_index = BM25Okapi(tokenized_corpus)
                    print(f"✅ BM25 index rebuilt with {len(self.documents_list)} documents")
        except Exception as e:
            print(f"⚠️ Could not rebuild BM25 index: {e}")
            self.bm25_index = None
            self.documents_list = []
    
    def add_documents(self, documents: List[Dict]):
        """Add document chunks to the vector database."""
        if not documents:
            print("⚠️ No documents to add!")
            return
        
        print(f"📝 Adding {len(documents)} chunks to database...")
        
        # Extract data
        texts = [doc['text'] for doc in documents]
        metadatas = [
            {
                'source': doc['source'], 
                'chunk_id': doc['chunk_id'],
                **doc.get('metadata', {})
            } 
            for doc in documents
        ]
        
        # Generate unique IDs
        ids = [
            hashlib.md5(f"{doc['source']}_{doc['chunk_id']}_{doc['text'][:50]}".encode()).hexdigest()
            for doc in documents
        ]
        
        # Create embeddings
        print("🔮 Creating embeddings (this may take a minute)...")
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32
        ).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Update BM25 index
        self.documents_list.extend(texts)
        self.document_metadata.extend(metadatas)
        tokenized_corpus = [doc.lower().split() for doc in self.documents_list]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        print(f"✅ Successfully added {len(documents)} chunks!")
        print(f"📊 Total chunks in database: {self.collection.count()}")
    
    def search_semantic(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Semantic search using embeddings.
        
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
            n_results=min(n_results, self.collection.count())
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                
                # Filter out very weak matches
                if distance > 1.5:
                    continue

                formatted_results.append({
                    'text': results['documents'][0][i],
                    'source': results['metadatas'][0][i].get('source', 'Unknown'),
                    'distance': distance,
                    'score': 1 - distance,  # Convert distance to similarity score
                    'metadata': results['metadatas'][0][i]
                })

        return formatted_results
    
    def search_bm25(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Keyword-based search using BM25.
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with BM25 scores
        """
        if not self.bm25_index or not self.documents_list:
            print("⚠️ BM25 index not available, using semantic search only")
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top results
        top_indices = np.argsort(bm25_scores)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    'text': self.documents_list[idx],
                    'source': self.document_metadata[idx].get('source', 'Unknown'),
                    'score': float(bm25_scores[idx]),
                    'metadata': self.document_metadata[idx]
                })
        
        return results
    
    def hybrid_search(self, query: str, n_results: int = 5, alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid search combining BM25 and semantic search.
        
        Args:
            query: The search query
            n_results: Number of results to return
            alpha: Weight for semantic search (1-alpha for BM25)
                  alpha=1.0 means pure semantic, alpha=0.0 means pure BM25
            
        Returns:
            List of relevant chunks with combined scores
        """
        # Get results from both methods (retrieve more for better fusion)
        semantic_results = self.search_semantic(query, n_results=n_results * 3)
        bm25_results = self.search_bm25(query, n_results=n_results * 3)
        
        # If one method fails, fall back to the other
        if not bm25_results:
            return semantic_results[:n_results]
        if not semantic_results:
            return bm25_results[:n_results]
        
        # Combine scores using weighted fusion
        combined_scores = {}
        
        # Normalize semantic scores
        max_sem_score = max([r['score'] for r in semantic_results]) if semantic_results else 1.0
        for result in semantic_results:
            text = result['text']
            norm_score = result['score'] / max_sem_score if max_sem_score > 0 else 0
            combined_scores[text] = {
                'semantic_score': norm_score,
                'bm25_score': 0,
                'result': result
            }
        
        # Normalize BM25 scores
        max_bm25_score = max([r['score'] for r in bm25_results]) if bm25_results else 1.0
        for result in bm25_results:
            text = result['text']
            norm_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0
            
            if text in combined_scores:
                combined_scores[text]['bm25_score'] = norm_score
            else:
                combined_scores[text] = {
                    'semantic_score': 0,
                    'bm25_score': norm_score,
                    'result': result
                }
        
        # Calculate final scores
        final_results = []
        for text, scores in combined_scores.items():
            combined_score = (alpha * scores['semantic_score'] + 
                            (1 - alpha) * scores['bm25_score'])
            
            result = scores['result'].copy()
            result['combined_score'] = combined_score
            result['semantic_score'] = scores['semantic_score']
            result['bm25_score'] = scores['bm25_score']
            final_results.append(result)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return final_results[:n_results]
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank documents using cross-encoder for better precision.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return after reranking
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc['text']] for doc in documents]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Add rerank scores to documents
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(rerank_scores[i])
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]
    
    def search_with_rerank(self, query: str, n_results: int = 5, 
                          initial_results: int = 20, alpha: float = 0.5) -> List[Dict]:
        """
        Complete search pipeline: hybrid search + reranking.
        
        Args:
            query: The search query
            n_results: Final number of results to return
            initial_results: Number of candidates to retrieve before reranking
            alpha: Weight for semantic search in hybrid search
            
        Returns:
            Reranked list of top results
        """
        # Step 1: Hybrid search to get candidates
        candidates = self.hybrid_search(query, n_results=initial_results, alpha=alpha)
        
        if not candidates:
            return []
        
        # Step 2: Rerank for precision
        final_results = self.rerank(query, candidates, top_k=n_results)
        
        return final_results
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'total_chunks': self.collection.count(),
            'collection_name': self.collection.name,
            'bm25_indexed': len(self.documents_list),
            'has_reranker': self.reranker is not None
        }
    
    def reset(self):
        """Delete all data and start fresh (explicit user action only)."""
        print("🗑️ Resetting database...")
        self.client.delete_collection("knowledge_base")
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"description": "Document embeddings with hybrid search"}
        )
        self.bm25_index = None
        self.documents_list = []
        self.document_metadata = []
        print("✅ Database reset complete!")
    
    def export_data(self, output_path: str = "./vector_store_backup.json"):
        """Export database for backup."""
        results = self.collection.get(include=['documents', 'metadatas'])
        
        data = {
            'documents': results['documents'],
            'metadatas': results['metadatas'],
            'count': len(results['documents'])
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported {data['count']} documents to {output_path}")
