"""Embedding models for RAG (Retrieval-Augmented Generation)"""
import os
import sys

# Disable TensorFlow to avoid conflicts with transformers
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

from sentence_transformers import SentenceTransformer
import numpy as np

# Add parent directory to path for config imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION

# Global variable to cache the model
_embedding_model = None

def get_embedding_model():
    """Get or initialize the embedding model (singleton pattern)"""
    global _embedding_model
    try:
        if _embedding_model is None:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return _embedding_model
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {str(e)}")

def get_embeddings(texts):
    """Generate embeddings for a list of texts
    
    Args:
        texts: List of strings or single string to embed
        
    Returns:
        numpy array of embeddings
    """
    try:
        model = get_embedding_model()
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings
    
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

def get_embedding_dimension():
    """Get the dimension of the embedding model"""
    return EMBEDDING_DIMENSION
