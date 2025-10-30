"""LLM Models for multiple providers (OpenAI, Groq, Gemini, Perplexity, DeepSeek, OpenRouter)"""
import os
import sys
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Add parent directory to path for config imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.config import (
    OPENAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, PERPLEXITY_API_KEY,
    DEEPSEEK_API_KEY, OPENROUTER_API_KEY, API_KEY_LINKS,
    DEFAULT_OPENAI_MODEL, DEFAULT_GROQ_MODEL, DEFAULT_GEMINI_MODEL,
    DEFAULT_PERPLEXITY_MODEL, DEFAULT_DEEPSEEK_MODEL, DEFAULT_OPENROUTER_MODEL
)

def get_openai_model(model_name=None, temperature=0.7, max_tokens=None, api_key=None):
    """Initialize and return the OpenAI chat model"""
    try:
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OpenAI API key not found in configuration")
        
        model = ChatOpenAI(
            api_key=key,
            model=model_name or DEFAULT_OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI model: {str(e)}")

def get_chatgroq_model(model_name=None, temperature=0.7, max_tokens=None, api_key=None):
    """Initialize and return the Groq chat model"""
    try:
        key = api_key or GROQ_API_KEY
        if not key:
            raise ValueError("Groq API key not found in configuration")
        
        model = ChatGroq(
            api_key=key,
            model=model_name or DEFAULT_GROQ_MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")

def get_gemini_model(model_name=None, temperature=0.7, max_tokens=None, api_key=None):
    """Initialize and return the Google Gemini chat model"""
    try:
        key = api_key or GOOGLE_API_KEY
        if not key:
            raise ValueError("Google API key not found in configuration")
        
        model = ChatGoogleGenerativeAI(
            google_api_key=key,
            model=model_name or DEFAULT_GEMINI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")

def get_perplexity_model(model_name=None, temperature=0.7, max_tokens=None, api_key=None):
    """Initialize and return the Perplexity chat model"""
    try:
        key = api_key or PERPLEXITY_API_KEY
        if not key:
            raise ValueError("Perplexity API key not found in configuration")
        
        # Perplexity uses OpenAI-compatible API
        model = ChatOpenAI(
            api_key=key,
            model=model_name or DEFAULT_PERPLEXITY_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url="https://api.perplexity.ai"
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Perplexity model: {str(e)}")

def get_deepseek_model(model_name=None, temperature=0.7, max_tokens=None, api_key=None):
    """Initialize and return the DeepSeek chat model"""
    try:
        key = api_key or DEEPSEEK_API_KEY
        if not key:
            raise ValueError("DeepSeek API key not found in configuration")
        
        # DeepSeek uses OpenAI-compatible API
        model = ChatOpenAI(
            api_key=key,
            model=model_name or DEFAULT_DEEPSEEK_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url="https://api.deepseek.com"
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize DeepSeek model: {str(e)}")

def get_openrouter_model(model_name=None, temperature=0.7, max_tokens=None, api_key=None):
    """Initialize and return the OpenRouter chat model"""
    try:
        key = api_key or OPENROUTER_API_KEY
        if not key:
            raise ValueError("OpenRouter API key not found in configuration")
        
        # OpenRouter uses OpenAI-compatible API
        model = ChatOpenAI(
            api_key=key,
            model=model_name or DEFAULT_OPENROUTER_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url="https://openrouter.ai/api/v1"
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenRouter model: {str(e)}")

def get_model(provider="groq", model_name=None, temperature=0.7, max_tokens=None, api_key=None):
    """Get a chat model based on provider
    
    Args:
        provider: One of 'openai', 'groq', 'gemini', 'perplexity', 'deepseek', or 'openrouter'
        model_name: Specific model name (optional)
        temperature: Model temperature (0-1)
        max_tokens: Maximum tokens in response
        api_key: API key for the provider (optional, will use config if not provided)
        
    Returns:
        Initialized chat model
    """
    try:
        provider = provider.lower()
        if provider == "openai":
            return get_openai_model(model_name, temperature, max_tokens, api_key)
        elif provider == "groq":
            return get_chatgroq_model(model_name, temperature, max_tokens, api_key)
        elif provider == "gemini":
            return get_gemini_model(model_name, temperature, max_tokens, api_key)
        elif provider == "perplexity":
            return get_perplexity_model(model_name, temperature, max_tokens, api_key)
        elif provider == "deepseek":
            return get_deepseek_model(model_name, temperature, max_tokens, api_key)
        elif provider == "openrouter":
            return get_openrouter_model(model_name, temperature, max_tokens, api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Choose from 'openai', 'groq', 'gemini', 'perplexity', 'deepseek', or 'openrouter'")
    except Exception as e:
        raise RuntimeError(f"Failed to get model: {str(e)}")

def get_available_providers():
    """Check which providers have API keys configured"""
    providers = []
    if OPENAI_API_KEY:
        providers.append("openai")
    if GROQ_API_KEY:
        providers.append("groq")
    if GOOGLE_API_KEY:
        providers.append("gemini")
    if PERPLEXITY_API_KEY:
        providers.append("perplexity")
    if DEEPSEEK_API_KEY:
        providers.append("deepseek")
    if OPENROUTER_API_KEY:
        providers.append("openrouter")
    return providers

def validate_api_key(provider, api_key):
    """Validate an API key by attempting to initialize the model
    
    Args:
        provider: The LLM provider name
        api_key: The API key to validate
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        # Try to create a model with the provided API key
        model = get_model(provider=provider, api_key=api_key, temperature=0.1, max_tokens=10)
        # If we can create the model, the key is valid
        return (True, "✅ API key is valid!")
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
            return (False, "❌ Invalid API key")
        else:
            return (False, f"❌ Error: {str(e)}")

def get_provider_info():
    """Get information about all available providers"""
    return {
        "openai": {
            "name": "OpenAI",
            "description": "GPT-4 and GPT-3.5 models",
            "link": API_KEY_LINKS["openai"],
            "free_tier": False,
            "recommended": True
        },
        "groq": {
            "name": "Groq",
            "description": "Ultra-fast Llama models (FREE)",
            "link": API_KEY_LINKS["groq"],
            "free_tier": True,
            "recommended": True
        },
        "gemini": {
            "name": "Google Gemini",
            "description": "Fast & high-quality (2.0/2.5 Flash/Pro) - FREE",
            "link": API_KEY_LINKS["gemini"],
            "free_tier": True,
            "recommended": True
        },
        "perplexity": {
            "name": "Perplexity",
            "description": "Online models with web search",
            "link": API_KEY_LINKS["perplexity"],
            "free_tier": False,
            "recommended": False
        },
        "deepseek": {
            "name": "DeepSeek",
            "description": "High-performance reasoning models",
            "link": API_KEY_LINKS["deepseek"],
            "free_tier": True,
            "recommended": True
        },
        "openrouter": {
            "name": "OpenRouter",
            "description": "Access to multiple models (FREE options)",
            "link": API_KEY_LINKS["openrouter"],
            "free_tier": True,
            "recommended": True
        }
    }