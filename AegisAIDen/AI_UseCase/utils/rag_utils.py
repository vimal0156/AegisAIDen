"""RAG (Retrieval-Augmented Generation) utilities for document processing and retrieval"""
import os
import sys
import pickle
import numpy as np
from typing import List, Tuple
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, VECTOR_STORE_PATH
from models.embeddings import get_embeddings

class DocumentChunk:
    """Represents a chunk of a document with its metadata"""
    def __init__(self, text, source, page=None, chunk_id=None):
        self.text = text
        self.source = source
        self.page = page
        self.chunk_id = chunk_id
        self.embedding = None

class VectorStore:
    """Simple vector store for document embeddings"""
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store"""
        try:
            self.chunks.extend(chunks)
            
            # Generate embeddings for new chunks
            texts = [chunk.text for chunk in chunks]
            new_embeddings = get_embeddings(texts)
            
            # Update embeddings array
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
                
        except Exception as e:
            raise RuntimeError(f"Failed to add chunks to vector store: {str(e)}")
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks using cosine similarity"""
        try:
            if not self.chunks or self.embeddings is None:
                return []
            
            # Get query embedding
            query_embedding = get_embeddings(query)
            
            # Calculate cosine similarity
            similarities = self._cosine_similarity(query_embedding, self.embeddings)
            
            # Get top-k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = [(self.chunks[i], float(similarities[i])) for i in top_indices]
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to search vector store: {str(e)}")
    
    def _cosine_similarity(self, query_vec, doc_vecs):
        """Calculate cosine similarity between query and document vectors"""
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        return np.dot(doc_norms, query_norm.T).flatten()
    
    def save(self, path: str):
        """Save vector store to disk"""
        try:
            if not path:
                raise ValueError("Path cannot be empty")
            
            # Ensure parent directory exists
            parent_dir = os.path.dirname(path)
            if parent_dir:  # Only create if there's a parent directory
                os.makedirs(parent_dir, exist_ok=True)
            
            with open(path, 'wb') as f:
                pickle.dump({'chunks': self.chunks, 'embeddings': self.embeddings}, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save vector store: {str(e)}")
    
    def load(self, path: str):
        """Load vector store from disk"""
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data['chunks']
                    self.embeddings = data['embeddings']
                return True
            return False
        except Exception as e:
            raise RuntimeError(f"Failed to load vector store: {str(e)}")
    
    def clear(self):
        """Clear all chunks and embeddings"""
        self.chunks = []
        self.embeddings = None

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    try:
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
            
        return chunks
    except Exception as e:
        raise RuntimeError(f"Failed to chunk text: {str(e)}")

def process_text_file(file_path: str, source_name: str = None) -> List[DocumentChunk]:
    """Process a text file and return document chunks"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        source = source_name or os.path.basename(file_path)
        text_chunks = chunk_text(text)
        
        doc_chunks = []
        for i, text_chunk in enumerate(text_chunks):
            chunk = DocumentChunk(
                text=text_chunk,
                source=source,
                chunk_id=i
            )
            doc_chunks.append(chunk)
        
        return doc_chunks
    except Exception as e:
        raise RuntimeError(f"Failed to process text file: {str(e)}")

def process_pdf_file(file_path: str, source_name: str = None) -> List[DocumentChunk]:
    """Process a PDF file and return document chunks"""
    try:
        import PyPDF2
        
        doc_chunks = []
        source = source_name or os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_chunks = chunk_text(text)
                
                for i, text_chunk in enumerate(text_chunks):
                    chunk = DocumentChunk(
                        text=text_chunk,
                        source=source,
                        page=page_num + 1,
                        chunk_id=f"{page_num}_{i}"
                    )
                    doc_chunks.append(chunk)
        
        return doc_chunks
    except ImportError:
        raise RuntimeError("PyPDF2 not installed. Install with: pip install PyPDF2")
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF file: {str(e)}")

def process_document(file_path: str, source_name: str = None) -> List[DocumentChunk]:
    """Process a document file (auto-detect type)"""
    try:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return process_pdf_file(file_path, source_name)
        elif file_ext in ['.txt', '.md', '.csv']:
            return process_text_file(file_path, source_name)
        else:
            # Try as text file
            return process_text_file(file_path, source_name)
    except Exception as e:
        raise RuntimeError(f"Failed to process document: {str(e)}")

def format_rag_context(search_results: List[Tuple[DocumentChunk, float]]) -> str:
    """Format search results into context for the LLM"""
    try:
        if not search_results:
            return ""
        
        context_parts = ["Here is relevant information from the knowledge base:\n"]
        
        for i, (chunk, score) in enumerate(search_results, 1):
            source_info = f"[Source: {chunk.source}"
            if chunk.page:
                source_info += f", Page {chunk.page}"
            source_info += f", Relevance: {score:.2f}]"
            
            context_parts.append(f"\n{i}. {source_info}\n{chunk.text}\n")
        
        return "\n".join(context_parts)
    except Exception as e:
        raise RuntimeError(f"Failed to format RAG context: {str(e)}")
