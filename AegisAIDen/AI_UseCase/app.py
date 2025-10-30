"""Medical Research Assistant - AI Chatbot with RAG and Web Search"""
import os
import sys

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import streamlit as st
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.llm import get_model, get_available_providers, validate_api_key, get_provider_info
from models.embeddings import get_embedding_model
from utils.rag_utils import VectorStore, process_document, format_rag_context
from utils.web_search import web_search, format_search_results, should_use_web_search
from config.config import (
    CONCISE_SYSTEM_PROMPT, DETAILED_SYSTEM_PROMPT,
    CONCISE_MAX_TOKENS, DETAILED_MAX_TOKENS,
    VECTOR_STORE_PATH, UPLOAD_FOLDER
)

# Initialize vector store globally
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
    # Try to load existing vector store
    try:
        st.session_state.vector_store.load(VECTOR_STORE_PATH)
    except:
        pass

def get_chat_response(chat_model, messages, system_prompt, use_rag=False, use_web_search=False, user_query="", provider=""):
    """Get response from the chat model with optional RAG and web search
    
    Args:
        chat_model: The LLM model to use
        messages: Conversation history
        system_prompt: System prompt for the model
        use_rag: Whether to use RAG for context
        use_web_search: Whether to use web search
        user_query: The current user query
        provider: The LLM provider name (to handle provider-specific formatting)
        
    Returns:
        Response string from the model
    """
    try:
        context_parts = []
        
        # Add RAG context if enabled
        if use_rag and st.session_state.vector_store.chunks:
            try:
                search_results = st.session_state.vector_store.search(user_query, top_k=3)
                if search_results:
                    rag_context = format_rag_context(search_results)
                    context_parts.append(rag_context)
            except Exception as e:
                st.warning(f"RAG search error: {str(e)}")
        
        # Add web search context if enabled
        if use_web_search:
            try:
                search_results = web_search(user_query, num_results=3)
                if search_results:
                    web_context = format_search_results(search_results)
                    context_parts.append(web_context)
            except Exception as e:
                st.warning(f"Web search error: {str(e)}")
    
        enhanced_prompt = system_prompt
        if context_parts:
            enhanced_prompt += "\n\n" + "\n\n".join(context_parts)
            enhanced_prompt += "\n\nPlease use the above information to answer the user's question accurately."
        
        # Prepare messages for the model
        formatted_messages = []
        
        # Gemini doesn't support SystemMessage, so we handle it differently
        if provider.lower() == "gemini":
            # For Gemini, we prepend the system prompt to EVERY user message to maintain context
            # Or we can just add it to the first message if there's history
            if len(messages) == 0:
                # First message - include system prompt
                formatted_messages.append(HumanMessage(content=f"{enhanced_prompt}\n\n{user_query}"))
            else:
                # Has history - add conversation history first
                for i, msg in enumerate(messages):
                    if msg["role"] == "user":
                        if i == 0:
                            # Prepend system prompt to first user message only
                            formatted_messages.append(HumanMessage(content=f"{enhanced_prompt}\n\n{msg['content']}"))
                        else:
                            formatted_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        formatted_messages.append(AIMessage(content=msg["content"]))
                
                # Add the current user query at the end
                formatted_messages.append(HumanMessage(content=user_query))
        else:
            # Other providers support SystemMessage
            formatted_messages = [SystemMessage(content=enhanced_prompt)]
            
            # Add conversation history
            for msg in messages:
                if msg["role"] == "user":
                    formatted_messages.append(HumanMessage(content=msg["content"]))
                else:
                    formatted_messages.append(AIMessage(content=msg["content"]))
            
            # Add current user query if provided
            if user_query and len(messages) > 0 and messages[-1]["content"] != user_query:
                formatted_messages.append(HumanMessage(content=user_query))
        
        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content
    
    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    """Instructions and setup page"""
    st.title("🏥 Medical Research Assistant")
    st.markdown("### AI-Powered Chatbot with RAG & Web Search")
    
    st.markdown("""
    ## 🎯 Use Case: Medical Research Assistant
    
    This intelligent chatbot helps medical professionals, researchers, and students:
    - **Query medical documents** using RAG (Retrieval-Augmented Generation)
    - **Get latest medical information** via real-time web search
    - **Switch between concise and detailed** response modes
    - **Support multiple LLM providers** (OpenAI, Groq, Gemini)
    
    ---
    
    ## 🔧 Installation
    
    Install required dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## 🔑 API Key Setup
    
    Set environment variables or edit `config/config.py`:
    
    ### LLM Providers (at least one required)
    - **OpenAI**: Get key from [OpenAI Platform](https://platform.openai.com/api-keys)
      ```bash
      set OPENAI_API_KEY=your_key_here
      ```
    
    - **Groq**: Get key from [Groq Console](https://console.groq.com/keys)
      ```bash
      set GROQ_API_KEY=your_key_here
      ```
    
    - **Google Gemini**: Get key from [Google AI Studio](https://aistudio.google.com/app/apikey)
      ```bash
      set GOOGLE_API_KEY=your_key_here
      ```
    
    ### Web Search (optional)
    - **Serper API**: Get key from [Serper.dev](https://serper.dev)
    - **Tavily API**: Get key from [Tavily](https://tavily.com)
    - **DuckDuckGo**: No API key needed (free fallback)
    
    ---
    
    ## 📚 Features
    
    ### 1️⃣ RAG (Retrieval-Augmented Generation)
    - Upload medical documents (PDF, TXT, MD, CSV)
    - Automatic document chunking and embedding
    - Semantic search for relevant context
    - Cite sources in responses
    
    ### 2️⃣ Live Web Search
    - Real-time medical information
    - Latest research and news
    - Automatic detection of queries needing current data
    
    ### 3️⃣ Response Modes
    - **Concise**: Brief, summarized answers (2-3 sentences)
    - **Detailed**: Comprehensive explanations with context
    
    ### 4️⃣ Multi-Provider Support
    - Switch between OpenAI, Groq, and Gemini
    - Automatic provider detection
    - Fallback options
    
    ---
    
    ## 🚀 How to Use
    
    1. **Upload Documents** (Document Management page)
       - Add medical papers, research articles, or textbooks
       - Documents are processed and indexed automatically
    
    2. **Configure Settings** (Sidebar)
       - Select LLM provider and model
       - Choose response mode (Concise/Detailed)
       - Enable/disable RAG and web search
    
    3. **Start Chatting** (Chat page)
       - Ask medical questions
       - Get context-aware responses
       - View sources and citations
    
    ---
    
    ## 💡 Example Queries
    
    - "What are the latest treatments for Type 2 diabetes?"
    - "Explain the mechanism of action of ACE inhibitors"
    - "What does my uploaded research paper say about COVID-19 vaccines?"
    - "Recent news on Alzheimer's disease research"
    
    ---
    
    ## 🛠️ Troubleshooting
    
    - **No API Keys Found**: Set at least one LLM provider API key
    - **RAG Not Working**: Upload documents in Document Management
    - **Web Search Failing**: Check internet connection or API keys
    - **Slow Responses**: Try switching to a faster model or provider
    
    ---
    
    Ready to start? Navigate to **Chat** or **Document Management** using the sidebar!
    """)
    
def document_management_page():
    """Document management page for RAG"""
    st.title("📚 Document Management")
    st.markdown("Upload and manage documents for RAG-based question answering")
    
    # Create upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # File upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, TXT, MD, CSV)",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Uploaded Files"):
            with st.spinner("Processing documents..."):
                try:
                    for uploaded_file in uploaded_files:
                        # Save file
                        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        chunks = process_document(file_path, uploaded_file.name)
                        st.session_state.vector_store.add_chunks(chunks)
                        
                        st.success(f"✅ Processed {uploaded_file.name} ({len(chunks)} chunks)")
                    
                    # Save vector store
                    st.session_state.vector_store.save(VECTOR_STORE_PATH)
                    st.success("🎉 All documents processed and indexed!")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    # Display current documents
    st.subheader("Current Knowledge Base")
    if st.session_state.vector_store.chunks:
        # Get unique sources
        sources = set(chunk.source for chunk in st.session_state.vector_store.chunks)
        
        st.info(f"📊 **Total Documents**: {len(sources)} | **Total Chunks**: {len(st.session_state.vector_store.chunks)}")
        
        # Display sources
        for source in sorted(sources):
            source_chunks = [c for c in st.session_state.vector_store.chunks if c.source == source]
            st.write(f"- **{source}** ({len(source_chunks)} chunks)")
        
        # Clear button
        if st.button("🗑️ Clear All Documents", type="secondary"):
            st.session_state.vector_store.clear()
            try:
                os.remove(VECTOR_STORE_PATH)
            except:
                pass
            st.success("All documents cleared!")
            st.rerun()
    else:
        st.warning("No documents in knowledge base. Upload documents above to get started.")

def chat_page():
    """Main chat interface page"""
    st.title("Medical Research Assistant")
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("Configuration")
        
        # Get provider information
        provider_info = get_provider_info()
        all_providers = list(provider_info.keys())
        
        # Provider selection with better display
        provider_display = {
            p: provider_info[p]['name']
            for p in all_providers
        }
        
        provider = st.selectbox(
            "Select LLM Provider",
            all_providers,
            format_func=lambda x: provider_display[x],
            help="Choose your AI provider"
        )
        
        # Display provider info
        st.caption(f"{provider_info[provider]['description']}")
        
        # API Key input and validation
        st.markdown("---")
        st.markdown("**API Key**")
        
        # Check if API key exists in session state
        if f'{provider}_api_key' not in st.session_state:
            st.session_state[f'{provider}_api_key'] = ""
        if f'{provider}_key_valid' not in st.session_state:
            st.session_state[f'{provider}_key_valid'] = False
        
        # API key input
        api_key_input = st.text_input(
            f"Enter {provider_info[provider]['name']} API Key",
            value=st.session_state[f'{provider}_api_key'],
            type="password",
            help=f"Get your API key from {provider_info[provider]['name']}",
            key=f"api_input_{provider}"
        )
        
        # Show link to get API key
        st.markdown(f"[Get API Key]({provider_info[provider]['link']})", unsafe_allow_html=True)
        
        # Validate button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Validate", use_container_width=True, key=f"validate_{provider}"):
                if api_key_input:
                    with st.spinner("Validating..."):
                        is_valid, message = validate_api_key(provider, api_key_input)
                        if is_valid:
                            st.session_state[f'{provider}_api_key'] = api_key_input
                            st.session_state[f'{provider}_key_valid'] = True
                            st.success(message)
                        else:
                            st.session_state[f'{provider}_key_valid'] = False
                            st.error(message)
                else:
                    st.warning("Please enter an API key")
        
        with col2:
            if st.button("Clear", use_container_width=True, key=f"clear_{provider}"):
                st.session_state[f'{provider}_api_key'] = ""
                st.session_state[f'{provider}_key_valid'] = False
                st.rerun()
        
        # Show validation status
        if st.session_state[f'{provider}_key_valid']:
            st.success("API Key Validated")
            current_api_key = st.session_state[f'{provider}_api_key']
        elif api_key_input:
            st.info("Click 'Validate' to verify your API key")
            current_api_key = None
        else:
            # Check if key exists in environment
            available_providers = get_available_providers()
            if provider in available_providers:
                st.success("Using API key from environment")
                current_api_key = None
            else:
                st.warning("No API key configured")
                current_api_key = None
        
        st.markdown("---")
        
        # SerpAPI Key (Optional for better web search)
        st.markdown("**SerpAPI Key (Optional)**")
        st.caption("For better web search quality. Leave empty to use DuckDuckGo (free).")
        
        serpapi_key = st.text_input(
            "Enter SerpAPI Key",
            type="password",
            help="Get free API key from https://serpapi.com/ (100 searches/month)",
            key="serpapi_key_input"
        )
        
        if serpapi_key:
            os.environ["SERPAPI_API_KEY"] = serpapi_key
            st.success("SerpAPI key configured")
        else:
            st.info("Using DuckDuckGo (free, unlimited)")
        
        st.markdown(f"[Get Free SerpAPI Key](https://serpapi.com/)")
        
        st.markdown("---")
        
        # Model selection based on provider (Latest & Most Powerful)
        model_options = {
            "openai": [
                "gpt-4o",  # Latest GPT-4 Omni (most powerful)
                "gpt-4-turbo",  # GPT-4 Turbo
                "gpt-4",  # GPT-4 (stable)
                "gpt-4o-mini",  # Fast & efficient
                "gpt-3.5-turbo"  # Legacy
            ],
            "groq": [
                "llama-3.3-70b-versatile",  # Latest Llama 3.3 (most powerful)
                "llama-3.1-70b-versatile",  # Llama 3.1 70B
                "mixtral-8x7b-32768",  # Mixtral 8x7B
                "llama-3.1-8b-instant",  # Fast 8B
                "gemma2-9b-it"  # Google Gemma 2
            ],
            "gemini": [
                "gemini-2.0-flash-exp",  # Latest Gemini 2.0 Flash (fastest)
                "gemini-2.0-flash-thinking-exp",  # Gemini 2.0 with reasoning
                "gemini-exp-1206",  # Experimental latest
                "gemini-1.5-pro-latest",  # Gemini 1.5 Pro (most capable)
                "gemini-1.5-flash-latest",  # Gemini 1.5 Flash
                "gemini-1.5-pro",  # Stable Pro
                "gemini-1.5-flash",  # Stable Flash
            ],
            "perplexity": [
                "llama-3.1-sonar-huge-128k-online",  # Largest model with web search
                "llama-3.1-sonar-large-128k-online",  # Large with web search
                "llama-3.1-sonar-small-128k-online"  # Small with web search
            ],
            "deepseek": [
                "deepseek-chat",  # Latest DeepSeek Chat
                "deepseek-reasoner"  # DeepSeek with reasoning
            ],
            "openrouter": [
                "anthropic/claude-3.5-sonnet",  # Claude 3.5 Sonnet (best)
                "google/gemini-pro-1.5",  # Gemini Pro 1.5
                "openai/gpt-4-turbo",  # GPT-4 Turbo
                "meta-llama/llama-3.1-405b-instruct",  # Llama 3.1 405B (largest)
                "meta-llama/llama-3.1-70b-instruct",  # Llama 3.1 70B
                "meta-llama/llama-3.1-8b-instruct:free",  # Free tier
                "google/gemini-flash-1.5:free",  # Free Gemini
            ]
        }
        
        selected_model = st.selectbox(
            "Select Model",
            model_options.get(provider, []),
            help="Choose the specific model to use"
        )
        
        # Response mode
        response_mode = st.radio(
            "Response Mode",
            ["Concise", "Detailed"],
            help="Concise: Brief answers | Detailed: Comprehensive explanations"
        )
        
        # RAG toggle
        use_rag = st.checkbox(
            "Enable RAG (Document Search)",
            value=True,
            help="Search uploaded documents for relevant context"
        )
        
        # Web search toggle
        use_web_search = st.checkbox(
            "Enable Web Search",
            value=False,
            help="Search the web for current information"
        )
        
        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        st.markdown("---")
        
        # Chat History Management
        st.markdown("**Chat History**")
        
        # Show message count
        message_count = len(st.session_state.get("messages", []))
        st.caption(f"Total messages: {message_count}")
        
        # View history button
        if st.button("View History", use_container_width=True):
            st.session_state.show_history = not st.session_state.get("show_history", False)
        
        # Clear history button
        if st.button("Clear History", use_container_width=True):
            if st.session_state.get("messages"):
                st.session_state.messages = []
                st.success("Chat history cleared")
                st.rerun()
            else:
                st.info("No history to clear")
    
    # Determine system prompt and max tokens based on mode
    if response_mode == "Concise":
        system_prompt = CONCISE_SYSTEM_PROMPT
        max_tokens = CONCISE_MAX_TOKENS
    else:
        system_prompt = DETAILED_SYSTEM_PROMPT
        max_tokens = DETAILED_MAX_TOKENS
    
    # Add medical context to system prompt
    system_prompt += "\n\nYou are a medical research assistant. Provide accurate, evidence-based information. Always cite sources when using RAG or web search results."
    
    # Initialize chat model
    try:
        chat_model = get_model(
            provider=provider,
            model_name=selected_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=current_api_key
        )
    except Exception as e:
        st.error(f"Failed to initialize model: {str(e)}")
        st.info("Please enter and validate your API key in the sidebar.")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display status
    status_cols = st.columns(5)
    with status_cols[0]:
        st.metric("Provider", provider.upper())
    with status_cols[1]:
        # Shorten model name if too long
        display_model = selected_model.split("/")[-1] if "/" in selected_model else selected_model
        display_model = display_model[:20] + "..." if len(display_model) > 20 else display_model
        st.metric("Model", display_model)
    with status_cols[2]:
        st.metric("Mode", response_mode)
    with status_cols[3]:
        rag_status = "ON" if (use_rag and st.session_state.vector_store.chunks) else "OFF"
        st.metric("RAG", rag_status)
    with status_cols[4]:
        web_status = "ON" if use_web_search else "OFF"
        st.metric("Web Search", web_status)
    
    st.divider()
    
    # Show chat history in expander if requested
    if st.session_state.get("show_history", False) and st.session_state.messages:
        with st.expander("Chat History", expanded=True):
            for idx, message in enumerate(st.session_state.messages):
                role_label = "You" if message["role"] == "user" else "Assistant"
                st.markdown(f"**{role_label} (Message {idx + 1}):**")
                st.markdown(message["content"])
                st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Auto-enable web search for certain queries
        auto_web_search = use_web_search or should_use_web_search(prompt)
        
        # Generate bot response (use all messages except the one we just added)
        with st.spinner("Thinking..."):
            response = get_chat_response(
                chat_model=chat_model,
                messages=st.session_state.messages[:-1],  # History before current question
                system_prompt=system_prompt,
                use_rag=use_rag,
                use_web_search=auto_web_search,
                user_query=prompt,
                provider=provider
            )
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to display the new messages
        st.rerun()

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Medical Research Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI with improved contrast
    st.markdown("""
        <style>
        /* Better contrast for metric boxes */
        .stMetric {
            background-color: #262730;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #404050;
        }
        .stMetric label {
            color: #FAFAFA !important;
            font-weight: 600;
            font-size: 14px;
        }
        .stMetric [data-testid="stMetricValue"] {
            color: #00D9FF !important;
            font-size: 20px;
            font-weight: bold;
        }
        
        /* Better button visibility */
        .stButton > button {
            background-color: #00D9FF;
            color: #0E1117;
            font-weight: 600;
            border: none;
            padding: 10px 24px;
            border-radius: 6px;
        }
        .stButton > button:hover {
            background-color: #00B8D4;
            color: #0E1117;
        }
        
        /* Input field improvements */
        .stTextInput > div > div > input {
            background-color: #1E1E1E;
            color: #FFFFFF;
            border: 1px solid #404050;
        }
        
        /* Selectbox improvements */
        .stSelectbox > div > div > div {
            background-color: #262730;
            color: #FFFFFF;
        }
        
        /* Success/Error message improvements */
        .stSuccess {
            background-color: #1B4D3E;
            color: #00FF9D;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #00FF9D;
        }
        .stError {
            background-color: #4D1B1B;
            color: #FF6B6B;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #FF6B6B;
        }
        .stWarning {
            background-color: #4D3D1B;
            color: #FFD93D;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #FFD93D;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Navigation
    with st.sidebar:
        st.title("🏥 Medical Assistant")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["💬 Chat", "📚 Document Management", "📖 Instructions"],
            index=0
        )
        
        st.markdown("---")
        
        # Add clear chat button for chat page
        if page == "💬 Chat":
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        # Footer
        st.markdown("---")
        st.caption("Built with Streamlit & LangChain")
        st.caption("NeoStats AI Engineer Use Case")
    
    # Route to appropriate page
    if page == "📖 Instructions":
        instructions_page()
    elif page == "📚 Document Management":
        document_management_page()
    elif page == "💬 Chat":
        chat_page()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check the Instructions page for setup help.")
