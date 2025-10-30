"""Configuration file for API keys and settings
DO NOT commit this file with actual API keys to version control
"""
import os

# LLM Provider API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Web Search API Keys
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")  # SerpAPI (FREE tier: 100 searches/month) - https://serpapi.com/
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")  # Serper (Paid) - https://serper.dev/
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # Tavily (Paid) - https://tavily.com/

# Model Configurations (Latest & Most Powerful)
DEFAULT_OPENAI_MODEL = "gpt-4o"  # Latest GPT-4 Omni (most powerful)
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"  # Latest Llama 3.3 70B
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-exp"  # Latest Gemini 2.0 Flash
DEFAULT_PERPLEXITY_MODEL = "llama-3.1-sonar-huge-128k-online"  # Largest with web search
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"  # Latest DeepSeek
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"  # Claude 3.5 Sonnet (best)

# API Key Links
API_KEY_LINKS = {
    "openai": "https://platform.openai.com/api-keys",
    "groq": "https://console.groq.com/keys",
    "gemini": "https://aistudio.google.com/app/apikey",
    "perplexity": "https://www.perplexity.ai/settings/api",
    "deepseek": "https://platform.deepseek.com/api_keys",
    "openrouter": "https://openrouter.ai/keys"
}

# Embedding Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free, local model
EMBEDDING_DIMENSION = 384

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 3

# Response Mode Settings (Optimized for Medical Research & RAG)
CONCISE_MAX_TOKENS = 400  # For quick FAQs and simple queries
DETAILED_MAX_TOKENS = 5000  # For RAG, PDFs, and long medical/technical reports

# System Prompts
CONCISE_SYSTEM_PROMPT = """You are a helpful medical research AI assistant. Provide concise, direct answers. 
Keep responses brief and to the point, typically 2-4 sentences. Focus on key findings and actionable information."""

DETAILED_SYSTEM_PROMPT = """You are a knowledgeable medical research AI assistant. Provide comprehensive, 
detailed responses with explanations, examples, clinical context, and evidence-based information when appropriate. 
For medical topics, include relevant research findings, treatment protocols, and cite sources when available."""

# Document Storage
import os as _os
from pathlib import Path as _Path

# Get the directory where this config file is located
_CONFIG_DIR = _Path(__file__).parent.parent
_DATA_DIR = _CONFIG_DIR / "data"

# Ensure data directory exists
_DATA_DIR.mkdir(exist_ok=True)

# Vector store file path (with .pkl extension)
VECTOR_STORE_PATH = str(_DATA_DIR / "vector_store.pkl")

# Upload folder path
UPLOAD_FOLDER = str(_DATA_DIR / "uploaded_documents")

# Ensure upload folder exists
_Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
