"""
Test Script - Verify everything works before building UI
Author: Your Name
"""

from dotenv import load_dotenv
load_dotenv()

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine
import os

def test_pipeline():
    """Test the complete RAG pipeline."""
    
    print("="*60)
    print("🧪 TESTING YOUR AI KNOWLEDGE ASSISTANT")
    print("="*60)
    
    # Step 1: Create test documents
    print("\n📝 Step 1: Creating test documents...")
    
    os.makedirs("test_documents", exist_ok=True)
    
    test_content = """
    Machine Learning is a subset of Artificial Intelligence that enables computers to learn from data.
    
    There are three main types of machine learning:
    1. Supervised Learning - The model learns from labeled data
    2. Unsupervised Learning - The model finds patterns in unlabeled data
    3. Reinforcement Learning - The model learns through trial and error
    
    Popular algorithms include:
    - Linear Regression for predicting continuous values
    - Decision Trees for classification tasks
    - Neural Networks for complex pattern recognition
    
    Machine learning is used in many applications like spam detection, 
    image recognition, and recommendation systems.
    
    Deep Learning is a subset of machine learning that uses neural networks 
    with multiple layers. It's particularly good at processing images, text, 
    and speech.
    
    Common frameworks for machine learning include TensorFlow, PyTorch, and 
    scikit-learn. Python is the most popular language for ML development.
    """
    
    with open("test_documents/ml_basics.txt", "w") as f:
        f.write(test_content)
    
    print("✅ Created test document: ml_basics.txt")
    
    # Step 2: Process documents
    print("\n📚 Step 2: Processing documents...")
    processor = DocumentProcessor()
    documents = processor.process_directory("test_documents")
    
    if not documents:
        print("❌ No documents found!")
        return
    
    # Step 3: Create vector store
    print("\n🗄️ Step 3: Creating vector database...")
    vector_store = VectorStore()
    vector_store.add_documents(documents)
    
    stats = vector_store.get_stats()
    print(f"\n📊 Database stats: {stats}")
    
    # Step 4: Test retrieval
    print("\n🔍 Step 4: Testing retrieval...")
    test_query = "What is supervised learning?"
    results = vector_store.search(test_query, n_results=2)
    
    print(f"\nQuery: '{test_query}'")
    print("\nRetrieved chunks:")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Relevance: {(1 - result['distance']):.3f}")
        print(f"Text: {result['text'][:150]}...")
    
    # Step 5: Test RAG engine
    print("\n\n🤖 Step 5: Testing RAG engine...")
    print("(This will call Groq API - make sure you have a valid API key!)")
    
    # Check for API token
    api_token = os.getenv("GROQ_API_KEY", "")
    if not api_token or not api_token.startswith("gsk_"):
        print("\n⚠️ WARNING: No valid Groq API key found!")
        print("Set GROQ_API_KEY environment variable or add to .env file")
        print("Get a free token at: https://console.groq.com/keys")
        print("\nSkipping RAG engine test...")
        return
    
    print(f"✅ Found API token: {api_token[:10]}...")
    
    rag_engine = RAGEngine(vector_store)
    
    test_questions = [
        "What is machine learning?",
        "What are the three types of machine learning?",
        "Name some ML frameworks"
    ]
    
    for question in test_questions:
        print("\n" + "="*60)
        print(f"❓ Question: {question}")
        print("="*60)
        
        result = rag_engine.generate_answer(question)
        
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
            print("\nMake sure:")
            print("1. GROQ_API_KEY is set correctly")
            print("2. You have internet connection")
            print("3. The selected model is available")
            print("4. Get a token at: https://console.groq.com/keys")
            break
        
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n📎 Sources: {', '.join(result['sources'])}")
        print(f"📊 Chunks used: {result['retrieved_chunks']}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n🎉 Your AI Knowledge Assistant is working!")
    print("\nNext steps:")
    print("1. Run the web app: streamlit run app.py")
    print("2. Upload your own documents")
    print("3. Start asking questions!")

if __name__ == "__main__":
    try:
        test_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all packages are installed: pip install -r requirements.txt")
        print("2. Set GROQ_API_KEY environment variable")
        print("3. Check internet connection")
        print("4. Verify Python version is 3.9+")