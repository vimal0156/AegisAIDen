# Features Documentation

## Overview

The Medical Research Assistant implements core features including RAG integration, web search, and configurable response modes.

---

## Core Features

### 1. RAG Integration (Retrieval-Augmented Generation)

#### Implementation Details

**Document Upload & Processing**
- **Supported Formats**: PDF, TXT, MD, CSV
- **Upload Interface**: Streamlit file uploader with multi-file support
- **Storage**: Local filesystem in `uploaded_documents/` directory

**Text Chunking**
- **Algorithm**: Sliding window with overlap
- **Chunk Size**: 1000 characters (configurable)
- **Overlap**: 200 characters (configurable)
- **Purpose**: Maintains context while keeping chunks manageable

**Embedding Generation**
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Type**: Dense vector embeddings
- **Advantages**: 
  - Free and runs locally
  - Fast inference
  - Good semantic understanding
  - No API costs

**Vector Storage**
- **Implementation**: Custom VectorStore class
- **Storage Format**: Pickle serialization
- **Persistence**: Saved to `vector_store` file
- **Session Management**: Loaded on app startup

**Semantic Search**
- **Algorithm**: Cosine similarity
- **Top-K Retrieval**: 3 most relevant chunks (configurable)
- **Scoring**: Normalized similarity scores (0-1)
- **Context Injection**: Results formatted and added to LLM prompt

**Code Location**
```
models/embeddings.py     - Embedding model management
utils/rag_utils.py       - Document processing, chunking, vector search
app.py                   - Document management UI and RAG integration
```

**Example Usage Flow**
1. User uploads medical research paper (PDF)
2. System extracts text and splits into chunks
3. Each chunk is embedded using sentence transformers
4. Embeddings stored in vector database
5. User asks: "What does the paper say about diabetes?"
6. Query is embedded and compared to stored chunks
7. Top 3 relevant chunks retrieved
8. Chunks added to LLM context
9. LLM generates response with citations

---

### 2. Live Web Search Integration

#### Implementation Details

**Search Providers**

**Primary: Serper API**
- **Type**: Google Search API
- **API Key**: Required (free tier: 2,500 searches/month)
- **Endpoint**: `https://google.serper.dev/search`
- **Response Time**: 2-3 seconds
- **Advantages**: High-quality Google results, structured data

**Secondary: Tavily API**
- **Type**: AI-optimized search
- **API Key**: Required
- **Endpoint**: `https://api.tavily.com/search`
- **Response Time**: 2-4 seconds
- **Advantages**: Optimized for LLM consumption

**Fallback: DuckDuckGo**
- **Type**: Free web search
- **API Key**: Not required
- **Library**: `duckduckgo-search`
- **Response Time**: 3-5 seconds
- **Advantages**: No API key needed, privacy-focused

**Smart Query Detection**
- **Keywords**: "latest", "recent", "current", "today", "news", "2024", "2025"
- **Auto-trigger**: Automatically enables web search for time-sensitive queries
- **Manual Override**: User can enable/disable via UI toggle

**Result Formatting**
- **Structure**: Title, snippet, source URL
- **Limit**: 3-5 results per query
- **Citation**: Results clearly marked with source links
- **Integration**: Formatted results added to LLM context

**Code Location**
```
utils/web_search.py      - All search provider implementations
config/config.py         - API key configuration
app.py                   - Web search UI controls
```

**Example Usage Flow**
1. User asks: "Latest diabetes treatment news 2024"
2. System detects "latest" and "2024" keywords
3. Auto-enables web search
4. Queries Serper API (or fallback to DuckDuckGo)
5. Retrieves top 3 results
6. Formats results with titles, snippets, URLs
7. Adds to LLM context
8. LLM generates response citing web sources

---

### 3. Response Modes: Concise vs Detailed

#### Implementation Details

**Concise Mode**
- **Max Tokens**: 150
- **Target Length**: 2-3 sentences
- **System Prompt**: "Provide brief, direct answers. Keep responses concise."
- **Use Cases**: Quick reference, mobile users, time-sensitive queries
- **Example**: 
  - Q: "What is diabetes?"
  - A: "Diabetes is a chronic disease where the body cannot properly regulate blood sugar due to insufficient insulin production or insulin resistance."

**Detailed Mode**
- **Max Tokens**: 1000
- **Target Length**: Multiple paragraphs
- **System Prompt**: "Provide comprehensive, detailed responses with explanations and context."
- **Use Cases**: Learning, research, complex topics
- **Example**:
  - Q: "What is diabetes?"
  - A: "Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels... [continues with pathophysiology, types, symptoms, treatment, etc.]"

**UI Implementation**
- **Control**: Radio button in sidebar
- **Options**: "Concise" or "Detailed"
- **Real-time**: Changes apply immediately to next query
- **Visual Feedback**: Current mode displayed in status metrics

**Technical Implementation**
```python
if response_mode == "Concise":
    system_prompt = CONCISE_SYSTEM_PROMPT
    max_tokens = CONCISE_MAX_TOKENS  # 150
else:
    system_prompt = DETAILED_SYSTEM_PROMPT
    max_tokens = DETAILED_MAX_TOKENS  # 1000
```

**Code Location**
```
config/config.py         - System prompts and token limits
app.py                   - Mode selection UI and logic
```

---

## Additional Features

### 4. Multi-Provider LLM Support

**Supported Providers**

**OpenAI**
- **Models**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Default**: gpt-4o-mini
- **Strengths**: High quality, reliable, well-documented
- **Cost**: ~$0.002/1K tokens (GPT-4o-mini)

**Groq**
- **Models**: Llama 3.1 70B, Llama 3.1 8B, Mixtral 8x7B
- **Default**: llama-3.1-70b-versatile
- **Strengths**: Extremely fast inference, generous free tier
- **Cost**: Free tier available

**Google Gemini**
- **Models**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **Default**: gemini-1.5-flash
- **Strengths**: Multimodal, large context window
- **Cost**: Free tier available

**Dynamic Provider Selection**
- **Auto-detection**: Checks which API keys are configured
- **UI Selection**: Dropdown in sidebar
- **Fallback**: If primary fails, can manually switch
- **Consistency**: Same features across all providers

**Code Location**
```
models/llm.py            - All provider implementations
app.py                   - Provider selection UI
```

---

### 5. Document Management Dashboard

**Features**
- **Upload Interface**: Drag-and-drop or file browser
- **Multi-file Support**: Upload multiple documents at once
- **Processing Status**: Real-time progress indicators
- **Document List**: View all uploaded documents
- **Chunk Statistics**: See chunk count per document
- **Clear Function**: Remove all documents and reset vector store

**UI Components**
- File uploader widget
- Process button with spinner
- Success/error messages
- Document statistics display
- Clear all button

---

### 6. Configuration Sidebar

**Settings Available**
- **LLM Provider**: OpenAI, Groq, or Gemini
- **Response Mode**: Concise or Detailed
- **RAG Toggle**: Enable/disable document search
- **Web Search Toggle**: Enable/disable web search
- **Temperature**: 0.0 to 1.0 (creativity control)

**Status Indicators**
- Current provider
- Current mode
- RAG status
- Web search status

---

### 7. Error Handling & Validation

**Comprehensive Try-Catch Blocks**
- All API calls wrapped in error handlers
- Graceful degradation on failures
- User-friendly error messages
- Logging for debugging

**Validation**
- API key presence checks
- File format validation
- Input sanitization
- Rate limit handling

**Examples**
```python
try:
    response = chat_model.invoke(messages)
    return response.content
except Exception as e:
    return f"Error getting response: {str(e)}"
```

---

### 8. Modern UI/UX

**Design Elements**
- Clean, medical-themed interface
- Emoji icons for visual appeal
- Color-coded status indicators
- Responsive layout
- Loading spinners for async operations

**User Experience**
- Intuitive navigation
- Clear instructions
- Real-time feedback
- Minimal clicks to functionality
- Mobile-friendly (Streamlit responsive design)

---

## Feature Comparison Matrix

| Feature | Status | Location | Configurable |
|---------|--------|----------|--------------|
| RAG Document Upload | Yes | Document Management | No |
| RAG Text Chunking | Yes | utils/rag_utils.py | Yes |
| RAG Embeddings | Yes | models/embeddings.py | Yes |
| RAG Vector Search | Yes | utils/rag_utils.py | Yes |
| Web Search (SerpAPI) | Yes | utils/web_search.py | Yes |
| Web Search (Serper) | Yes | utils/web_search.py | Yes |
| Web Search (Tavily) | Yes | utils/web_search.py | Yes |
| Web Search (DuckDuckGo) | Yes | utils/web_search.py | No |
| Concise Mode | Yes | app.py | Yes |
| Detailed Mode | Yes | app.py | Yes |
| OpenAI Support | Yes | models/llm.py | Yes |
| Groq Support | Yes | models/llm.py | Yes |
| Gemini Support | Yes | models/llm.py | Yes |
| Document Management UI | Yes | app.py | No |
| Configuration Sidebar | Yes | app.py | No |
| Error Handling | Yes | All files | No |
| Status Indicators | Yes | app.py | No |

---

## Configuration Options

All features are configurable via `config/config.py`:

```python
# RAG Configuration
CHUNK_SIZE = 1000              # Adjust chunk size
CHUNK_OVERLAP = 200            # Adjust overlap
TOP_K_RESULTS = 3              # Number of chunks to retrieve

# Response Modes
CONCISE_MAX_TOKENS = 150       # Concise response length
DETAILED_MAX_TOKENS = 1000     # Detailed response length

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

---

## Performance Metrics

**RAG Performance**
- Document processing: ~2-3 sec/page
- Embedding generation: ~0.5 sec/chunk
- Vector search: <1 second
- Total RAG overhead: 1-2 seconds

**Web Search Performance**
- Serper API: 2-3 seconds
- Tavily API: 2-4 seconds
- DuckDuckGo: 3-5 seconds

**LLM Response Time**
- Groq: 1-3 seconds (fastest)
- OpenAI: 2-5 seconds
- Gemini: 2-4 seconds

**Total Response Time**
- RAG only: 3-5 seconds
- Web search only: 4-7 seconds
- RAG + Web search: 6-10 seconds

---

## Usage Examples

### Example 1: RAG Query
```
User: "What does my uploaded paper say about Type 2 diabetes treatment?"

System:
1. Embeds query
2. Searches vector store
3. Retrieves relevant chunks from uploaded paper
4. Adds chunks to context
5. LLM generates response with citations

Response: "According to the uploaded document [diabetes_overview.txt], 
Type 2 diabetes management includes lifestyle modifications (diet and 
exercise) and Metformin as initial pharmacotherapy, with additional 
medications added as needed..."
```

### Example 2: Web Search Query
```
User: "Latest news on Alzheimer's disease treatment 2024"

System:
1. Detects "latest" and "2024" keywords
2. Triggers web search
3. Queries Serper/DuckDuckGo
4. Retrieves recent articles
5. Formats results
6. LLM synthesizes information

Response: "Recent developments in Alzheimer's treatment include... 
[Source: Nature Medicine, 2024] [Source: NIH News, 2024]"
```

### Example 3: Combined RAG + Web Search
```
User: "Compare the diabetes guidelines in my document with recent 
research findings"

System:
1. RAG: Retrieves guidelines from uploaded document
2. Web Search: Finds recent research articles
3. Combines both contexts
4. LLM compares and contrasts

Response: "Your uploaded guidelines recommend... Recent research 
published in 2024 suggests... Key differences include..."
```

---

## Future Enhancement Ideas

- Multi-document comparison
- Export conversation history
- Voice input/output
- Image analysis
- Custom embedding models
- Advanced vector databases
- Async processing
- Query caching
- User authentication
