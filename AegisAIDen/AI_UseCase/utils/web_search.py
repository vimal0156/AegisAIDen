"""Web search integration utilities"""
import os
import sys
import requests
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import SERPAPI_API_KEY, SERPER_API_KEY, TAVILY_API_KEY

def search_with_serpapi(query: str, num_results: int = 5) -> List[Dict]:
    """Search the web using SerpAPI (Google Search - FREE tier available)
    
    Get your free API key at: https://serpapi.com/
    Free tier: 100 searches/month
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search results with title, snippet, and link
    """
    try:
        if not SERPAPI_API_KEY:
            raise ValueError("SerpAPI key not configured")
        
        url = "https://serpapi.com/search"
        
        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": num_results,
            "engine": "google"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        if "organic_results" in data:
            for item in data["organic_results"][:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                })
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"SerpAPI search failed: {str(e)}")

def search_with_serper(query: str, num_results: int = 5) -> List[Dict]:
    """Search the web using Serper API (Google Search - Paid)
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search results with title, snippet, and link
    """
    try:
        if not SERPER_API_KEY:
            raise ValueError("Serper API key not configured")
        
        url = "https://google.serper.dev/search"
        
        payload = {
            "q": query,
            "num": num_results
        }
        
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        if "organic" in data:
            for item in data["organic"][:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                })
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Serper search failed: {str(e)}")

def search_with_tavily(query: str, num_results: int = 5) -> List[Dict]:
    """Search the web using Tavily API
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search results with title, snippet, and link
    """
    try:
        if not TAVILY_API_KEY:
            raise ValueError("Tavily API key not configured")
        
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": num_results,
            "search_depth": "basic"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        if "results" in data:
            for item in data["results"][:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "link": item.get("url", "")
                })
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Tavily search failed: {str(e)}")

def search_duckduckgo(query: str, num_results: int = 5) -> List[Dict]:
    """Search the web using DuckDuckGo (free, no API key required)
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search results with title, snippet, and link
    """
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=num_results)
            for item in search_results:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "link": item.get("href", "")
                })
        
        return results
        
    except ImportError:
        raise RuntimeError("duckduckgo-search not installed. Install with: pip install duckduckgo-search")
    except Exception as e:
        raise RuntimeError(f"DuckDuckGo search failed: {str(e)}")

def web_search(query: str, num_results: int = 5, provider: str = "auto") -> List[Dict]:
    """Perform web search using available provider
    
    Args:
        query: Search query
        num_results: Number of results to return
        provider: Search provider ('serper', 'tavily', 'duckduckgo', or 'auto')
        
    Returns:
        List of search results
    """
    try:
        # Auto-select provider based on available API keys
        if provider == "auto":
            if SERPAPI_API_KEY:
                provider = "serpapi"  # FREE tier - 100 searches/month
            elif SERPER_API_KEY:
                provider = "serper"
            elif TAVILY_API_KEY:
                provider = "tavily"
            else:
                provider = "duckduckgo"  # FREE - unlimited
        
        # Perform search with selected provider
        if provider == "serpapi":
            return search_with_serpapi(query, num_results)
        elif provider == "serper":
            return search_with_serper(query, num_results)
        elif provider == "tavily":
            return search_with_tavily(query, num_results)
        elif provider == "duckduckgo":
            return search_duckduckgo(query, num_results)
        else:
            raise ValueError(f"Unknown search provider: {provider}")
            
    except Exception as e:
        # Fallback to DuckDuckGo if primary provider fails
        if provider != "duckduckgo":
            try:
                return search_duckduckgo(query, num_results)
            except:
                pass
        raise RuntimeError(f"Web search failed: {str(e)}")

def format_search_results(results: List[Dict]) -> str:
    """Format search results into readable text for LLM context
    
    Args:
        results: List of search results
        
    Returns:
        Formatted string with search results
    """
    try:
        if not results:
            return "No search results found."
        
        formatted = ["Here are relevant web search results:\n"]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No description")
            link = result.get("link", "")
            
            formatted.append(f"\n{i}. **{title}**")
            formatted.append(f"   {snippet}")
            if link:
                formatted.append(f"   Source: {link}")
            formatted.append("")
        
        return "\n".join(formatted)
        
    except Exception as e:
        raise RuntimeError(f"Failed to format search results: {str(e)}")

def should_use_web_search(query: str) -> bool:
    """Determine if a query should trigger web search
    
    Args:
        query: User query
        
    Returns:
        True if web search should be used
    """
    # Keywords that suggest need for current information
    web_keywords = [
        "latest", "recent", "current", "today", "news", "update",
        "what is happening", "what's new", "2024", "2025",
        "weather", "stock", "price", "score"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in web_keywords)
