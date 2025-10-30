"""Quick test to verify all imports work correctly"""
import os
import sys

# Disable TensorFlow
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

print("Testing imports...")
print("=" * 50)

try:
    print("1. Testing embeddings module...")
    from models.embeddings import get_embedding_model
    print("   ✅ Embeddings module loaded successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

try:
    print("2. Testing LLM module...")
    from models.llm import get_model, get_available_providers
    print("   ✅ LLM module loaded successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

try:
    print("3. Testing RAG utilities...")
    from utils.rag_utils import VectorStore, process_document
    print("   ✅ RAG utilities loaded successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

try:
    print("4. Testing web search...")
    from utils.web_search import web_search
    print("   ✅ Web search module loaded successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

try:
    print("5. Testing config...")
    from config.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION
    print("   ✅ Config loaded successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

print("=" * 50)
print("✅ ALL IMPORTS SUCCESSFUL! App is ready to launch.")
print("=" * 50)
