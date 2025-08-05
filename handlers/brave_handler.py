import os
import time
import aiohttp
import logging
from typing import Dict, Any, Optional

from handlers.base_handler import ProviderHandler

logger = logging.getLogger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1"


class BraveHandler(ProviderHandler):
    """Handles interactions with the Brave Search API."""

    def __init__(
            self,
            search_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the BraveHandler.

        Args:
            search_params: Default search parameters to use for all searches
        """
        super().__init__(
            api_key=os.getenv("BRAVE_API_KEY"),
            api_url=BRAVE_API_URL,
            search_params=search_params
        )
        self.is_llm_response = False

    async def search(self, query: str) -> Dict[str, Any]:
        """Run a Brave search using async HTTP request.

        Args:
            inputs: Dictionary containing input data, must include 'question' key

        Returns:
            Dictionary containing 'answer' and 'search_response'
        """
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json"
        }

        params = {
            "q": query,
            **self.search_params
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{self.api_url}/web/search",
                        params=params,
                        headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Error in Brave search: HTTP {response.status}")
                        error_text = await response.text()
                        logger.error(f"Response: {error_text}")
                        return {
                            "answer": "",
                            "search_response": None
                        }

                    response_data = await response.json()
                    logger.info("Received response from Brave Search API")

                    return {
                        "answer": "",
                        "search_response": response_data
                    }

        except Exception as e:
            logger.error(f"Error in Brave search: {str(e)}")
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

        # Extract web results
        if "web" in response_data and "results" in response_data["web"]:
            for result in response_data["web"]["results"]:
                url = result.get("url", "")
                title = result.get("title", "")
                description = result.get("description", "")
                content = f"{title}\n{description}" if title and description else title or description
                if url and content:
                    search_results.append((url, content))

        return self._format_search_results_for_prompt(search_results)
