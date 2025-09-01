"""DuckDuckGo search tool implementation."""

import asyncio
from typing import Any, Dict, List

import structlog
from duckduckgo_search import DDGS

from app.tools.base import Tool, ToolResult
from app.utils.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class WebSearchTool(Tool):
    """Tool for searching the web using DuckDuckGo."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information using DuckDuckGo. Input: {'query': 'search terms'}"
        )
        self.max_results = 10
    
    async def execute(self, query: str, max_results: int = 10) -> ToolResult:
        """Execute web search."""
        try:
            # Use thread pool for blocking DuckDuckGo search
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                self._search_sync, 
                query, 
                min(max_results, self.max_results)
            )
            
            if not results:
                return ToolResult(
                    success=True,
                    result="No search results found.",
                    metadata={"query": query, "count": 0}
                )
            
            # Format results
            formatted_results = self._format_results(results)
            
            return ToolResult(
                success=True,
                result=formatted_results,
                metadata={"query": query, "count": len(results)}
            )
            
        except Exception as e:
            logger.error("Web search failed", query=query, error=str(e))
            return ToolResult(
                success=False,
                result="",
                error=f"Search failed: {str(e)}"
            )
    
    def _search_sync(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Synchronous search function for thread executor."""
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))
            return results
        except Exception as e:
            logger.error("DuckDuckGo search error", error=str(e))
            return []
    
    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into readable text."""
        formatted = "Search Results:\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            snippet = result.get('body', 'No description')
            url = result.get('href', 'No URL')
            
            formatted += f"{i}. **{title}**\n"
            formatted += f"   {snippet}\n"
            formatted += f"   URL: {url}\n\n"
        
        return formatted
