"""
Test script for Medical Research Assistant features
Run this to verify all components are working correctly
"""

import os
import sys

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("TEST 1: Checking Package Imports")
    print("=" * 60)
    
    packages = [
        ("streamlit", "Streamlit"),
        ("langchain_core", "LangChain Core"),
        ("langchain_openai", "LangChain OpenAI"),
        ("langchain_groq", "LangChain Groq"),
        ("langchain_google_genai", "LangChain Google GenAI"),
        ("sentence_transformers", "Sentence Transformers"),
        ("numpy", "NumPy"),
        ("PyPDF2", "PyPDF2"),
        ("requests", "Requests"),
    ]
    
    all_passed = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FAILED - {str(e)}")
            all_passed = False
    
    print()
    return all_passed

def test_config():
    """Test configuration file"""
    print("=" * 60)
    print("TEST 2: Checking Configuration")
    print("=" * 60)
    
    try:
        from config.config import (
            OPENAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY,
            EMBEDDING_MODEL, CHUNK_SIZE, TOP_K_RESULTS
        )
        
        print(f"✅ Config file loaded successfully")
        print(f"   - Embedding Model: {EMBEDDING_MODEL}")
        print(f"   - Chunk Size: {CHUNK_SIZE}")
        print(f"   - Top-K Results: {TOP_K_RESULTS}")
        
        # Check API keys
        api_keys_found = []
        if OPENAI_API_KEY:
            api_keys_found.append("OpenAI")
        if GROQ_API_KEY:
            api_keys_found.append("Groq")
        if GOOGLE_API_KEY:
            api_keys_found.append("Google")
        
        if api_keys_found:
            print(f"✅ API Keys configured: {', '.join(api_keys_found)}")
        else:
            print(f"⚠️  No API keys found - set at least one LLM provider key")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {str(e)}")
        print()
        return False

def test_embeddings():
    """Test embedding model"""
    print("=" * 60)
    print("TEST 3: Testing Embedding Model")
    print("=" * 60)
    
    try:
        from models.embeddings import get_embeddings, get_embedding_dimension
        
        # Test embedding generation
        test_texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = get_embeddings(test_texts)
        
        print(f"✅ Embedding model loaded successfully")
        print(f"   - Embedding dimension: {get_embedding_dimension()}")
        print(f"   - Generated embeddings shape: {embeddings.shape}")
        
        # Test single text
        single_embedding = get_embeddings("Single test")
        print(f"   - Single text embedding shape: {single_embedding.shape}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {str(e)}")
        print()
        return False

def test_rag_utils():
    """Test RAG utilities"""
    print("=" * 60)
    print("TEST 4: Testing RAG Utilities")
    print("=" * 60)
    
    try:
        from utils.rag_utils import (
            chunk_text, VectorStore, DocumentChunk
        )
        
        # Test text chunking
        test_text = "This is a test. " * 100
        chunks = chunk_text(test_text, chunk_size=50, overlap=10)
        print(f"✅ Text chunking: {len(chunks)} chunks created")
        
        # Test vector store
        vector_store = VectorStore()
        
        # Create test chunks
        doc_chunks = [
            DocumentChunk("Diabetes is a chronic disease.", "test.txt", chunk_id=0),
            DocumentChunk("Insulin regulates blood sugar.", "test.txt", chunk_id=1),
        ]
        
        vector_store.add_chunks(doc_chunks)
        print(f"✅ Vector store: {len(vector_store.chunks)} chunks added")
        
        # Test search
        results = vector_store.search("What is diabetes?", top_k=1)
        print(f"✅ Vector search: Found {len(results)} results")
        if results:
            print(f"   - Top result: '{results[0][0].text[:50]}...'")
            print(f"   - Similarity score: {results[0][1]:.3f}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ RAG utilities test failed: {str(e)}")
        print()
        return False

def test_web_search():
    """Test web search functionality"""
    print("=" * 60)
    print("TEST 5: Testing Web Search")
    print("=" * 60)
    
    try:
        from utils.web_search import search_duckduckgo, format_search_results
        
        # Test DuckDuckGo search (no API key needed)
        print("Testing DuckDuckGo search (may take a few seconds)...")
        results = search_duckduckgo("diabetes treatment", num_results=2)
        
        print(f"✅ Web search: Found {len(results)} results")
        if results:
            print(f"   - First result: {results[0].get('title', 'N/A')[:50]}...")
        
        # Test formatting
        formatted = format_search_results(results)
        print(f"✅ Result formatting: {len(formatted)} characters")
        
        print()
        return True
        
    except Exception as e:
        print(f"⚠️  Web search test failed: {str(e)}")
        print("   (This is OK if you don't have internet or duckduckgo-search installed)")
        print()
        return True  # Don't fail overall test

def test_llm_models():
    """Test LLM model initialization"""
    print("=" * 60)
    print("TEST 6: Testing LLM Models")
    print("=" * 60)
    
    try:
        from models.llm import get_available_providers
        
        providers = get_available_providers()
        
        if providers:
            print(f"✅ Available LLM providers: {', '.join(providers)}")
            
            # Try to initialize one model
            from models.llm import get_model
            try:
                model = get_model(provider=providers[0])
                print(f"✅ Successfully initialized {providers[0]} model")
            except Exception as e:
                print(f"⚠️  Model initialization warning: {str(e)}")
        else:
            print(f"⚠️  No LLM providers configured")
            print("   Set at least one API key to test LLM functionality")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ LLM model test failed: {str(e)}")
        print()
        return False

def test_document_processing():
    """Test document processing"""
    print("=" * 60)
    print("TEST 7: Testing Document Processing")
    print("=" * 60)
    
    try:
        from utils.rag_utils import process_text_file
        import tempfile
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Diabetes mellitus is a chronic disease. " * 50)
            temp_file = f.name
        
        try:
            # Process the file
            chunks = process_text_file(temp_file, "test_document.txt")
            print(f"✅ Document processing: {len(chunks)} chunks created")
            print(f"   - First chunk length: {len(chunks[0].text)} characters")
            print(f"   - Source: {chunks[0].source}")
        finally:
            # Clean up
            os.unlink(temp_file)
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Document processing test failed: {str(e)}")
        print()
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MEDICAL RESEARCH ASSISTANT - FEATURE TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Package Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Embedding Model", test_embeddings()))
    results.append(("RAG Utilities", test_rag_utils()))
    results.append(("Web Search", test_web_search()))
    results.append(("LLM Models", test_llm_models()))
    results.append(("Document Processing", test_document_processing()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Set API keys (if not already done)")
        print("2. Run: streamlit run app.py")
        print("3. Upload documents and start chatting!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Set API keys in environment variables or config.py")
        print("- Check internet connection for web search")
    
    print()

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
