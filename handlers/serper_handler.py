import os
import logging
import aiohttp
from typing import Dict, Any, Optional

from handlers.base_handler import ProviderHandler

logger = logging.getLogger(__name__)


SERPER_API_URL = "https://google.serper.dev"


class SerperHandler(ProviderHandler):
    """Handles interactions with the Serper API."""

    def __init__(
            self,
            search_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the SerperHandler.

        Args:
            search_params: Default search parameters to use for all searches
        """
        super().__init__(
            api_key=os.getenv("SERPER_API_KEY"),
            api_url=SERPER_API_URL,
            search_params=search_params
        )
        self.is_llm_response = False

    async def search(self, query: str) -> Dict[str, Any]:
        """Run a Serper search using async HTTP request.

        Args:
            query: The query to search for

        Returns:
            Dictionary containing 'answer' and 'search_response'
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "q": query,
            **self.search_params
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.api_url}/search",
                        json=payload,
                        headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Error in Serper search: HTTP {response.status}")
                        error_text = await response.text()
                        logger.error(f"Response: {error_text}")
                        return {
                            "answer": "",
                            "search_response": None
                        }

                    response_data = await response.json()
                    logger.info("Received response from Serper API")

                    return {
                        "answer": "",
                        "search_response": response_data
                    }

        except Exception as e:
            logger.error(f"Error in Serper search: {str(e)}")
            return {
                "answer": "",
                "search_response": None
            }

    async def post_process(self, search_response: dict) -> str:
        """
        Post process search response.

        Args:
            search_response: Dictionary containing the search response

        Returns:
            processed response ready for LLM prompt
        """
        if "search_response" not in search_response:
            return ""

        response_data = search_response["search_response"]
        search_results = []

        # Extract organic results
        if "organic" in response_data:
            for result in response_data["organic"]:
                url = result.get("link", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                content = f"{title}\n{snippet}" if title and snippet else title or snippet
                if url and content:
                    search_results.append((url, content))

        return self._format_search_results_for_prompt(search_results)
