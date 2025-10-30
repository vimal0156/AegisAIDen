# Streamlit Cloud Deployment Guide

## Quick Deploy Steps

1. **Go to Streamlit Cloud**: https://share.streamlit.io/

2. **Click "New app"**

3. **Configure your app**:
   - **Repository**: `vimal0156/AegisAIDen`
   - **Branch**: `main`
   - **Main file path**: `AegisAIDen/AI_UseCase/app.py`

4. **Add Secrets** (Click "Advanced settings" → "Secrets"):

```toml
# Add at least ONE LLM provider API key (required)
GROQ_API_KEY = "your_groq_api_key_here"
GOOGLE_API_KEY = "your_google_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"

# Optional: Additional providers
PERPLEXITY_API_KEY = "your_perplexity_key_here"
DEEPSEEK_API_KEY = "your_deepseek_key_here"
OPENROUTER_API_KEY = "your_openrouter_key_here"

# Optional: Web search providers
SERPER_API_KEY = "your_serper_key_here"
TAVILY_API_KEY = "your_tavily_key_here"
SERPAPI_API_KEY = "your_serpapi_key_here"
```

5. **Click "Deploy"**

## Get Free API Keys

### Required (at least one):
- **Groq** (FREE, recommended): https://console.groq.com/keys
- **Google Gemini** (FREE): https://aistudio.google.com/app/apikey
- **OpenAI** (Paid): https://platform.openai.com/api-keys

### Optional Web Search:
- **SerpAPI** (FREE tier): https://serpapi.com/
- **DuckDuckGo**: No API key needed (built-in fallback)

## Troubleshooting

### Import Errors
✅ **Fixed**: All import paths have been updated for Streamlit Cloud compatibility.

### Missing API Keys
- Add API keys in Streamlit Cloud "Secrets" section
- At least ONE LLM provider key is required
- Web search keys are optional (DuckDuckGo works without keys)

### Module Not Found
- Ensure `requirements.txt` is in `AegisAIDen/AI_UseCase/` directory
- Check that the main file path is correct: `AegisAIDen/AI_UseCase/app.py`

## Features Available

✅ RAG (Retrieval-Augmented Generation)
✅ Live Web Search
✅ Multiple LLM Providers
✅ Document Upload & Processing
✅ Concise & Detailed Response Modes

## Support

For issues, check the Streamlit Cloud logs:
- Click "Manage app" (bottom right)
- View logs for detailed error messages
