"""Web content fetching tool with HTML to markdown conversion."""

import asyncio
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import structlog
from markdownify import markdownify as md

from app.tools.base import Tool, ToolResult
from app.utils.config import get_settings
from app.utils.security import is_url_allowed

logger = structlog.get_logger()
settings = get_settings()


class WebFetchTool(Tool):
    """Tool for fetching web content and converting to markdown."""
    
    def __init__(self):
        super().__init__(
            name="web_fetch",
            description="Fetch content from a web URL and convert to markdown. Input: {'url': 'https://example.com'}"
        )
        self.timeout = 30
        self.max_content_length = 1024 * 1024  # 1MB
    
    async def execute(self, url: str) -> ToolResult:
        """Fetch web content."""
        # Validate URL
        if not self._is_valid_url(url):
            return ToolResult(
                success=False,
                result="",
                error="Invalid URL format"
            )
        
        # Check URL allowlist in safe mode
        if settings.SAFE_MODE and not is_url_allowed(url):
            return ToolResult(
                success=False,
                result="",
                error="URL not in allowlist"
            )
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(url) as response:
                    # Check response status
                    if response.status != 200:
                        return ToolResult(
                            success=False,
                            result="",
                            error=f"HTTP {response.status}: {response.reason}"
                        )
                    
                    # Check content length
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > self.max_content_length:
                        return ToolResult(
                            success=False,
                            result="",
                            error="Content too large"
                        )
                    
                    # Read content
                    html_content = await response.text()
                    
                    # Convert HTML to markdown
                    markdown_content = md(
                        html_content,
                        heading_style="ATX",
                        bullets="-",
                        strip=['script', 'style']
                    )
                    
                    # Truncate if too long
                    if len(markdown_content) > 50000:  # 50KB
                        markdown_content = markdown_content[:50000] + "\n\n[Content truncated...]"
                    
                    return ToolResult(
                        success=True,
                        result=markdown_content,
                        metadata={
                            "url": url,
                            "status_code": response.status,
                            "content_length": len(markdown_content)
                        }
                    )
                    
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                result="",
                error="Request timeout"
            )
        except Exception as e:
            logger.error("Web fetch failed", url=url, error=str(e))
            return ToolResult(
                success=False,
                result="",
                error=f"Fetch failed: {str(e)}"
            )
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except:
            return False

