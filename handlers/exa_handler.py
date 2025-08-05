import os
import time
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import aiohttp

from handlers.base_handler import ProviderHandler

load_dotenv()

logger = logging.getLogger(__name__)

EXA_API_URL = "https://api.exa.ai"


class ExaHandler(ProviderHandler):
    """
    Handles interactions with the Exa API.
    """

    def __init__(self, search_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the ExaHandler with the Exa API client.
        """
        super().__init__(
            api_key=os.getenv("EXA_API_KEY"),
            api_url=EXA_API_URL,
            search_params=search_params
        )
        self.is_llm_response = False

    async def search(self, query: str) -> Dict[str, Any]:
        """
        Run a Tavily search using async HTTP request.

        Args:
            query: The query to search for

        Returns:
            Dictionary containing 'answer' and 'search_response'
        """
        headers = {
            'content-type': 'application/json',
            'x-api-key': self.api_key,
        }

        # Construct request data
        data = {
            'query': query,
            **self.search_params
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.api_url}/search",
                        json=data,
                        headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Error in Exa search: HTTP {response.status}")
                        error_text = await response.text()
                        logger.error(f"Response: {error_text}")
                        return {
                            "answer": "",
                            "search_response": None
                        }

                    response_data = await response.json()
                    return {
                        "answer": response_data.get("answer", ""),
                        "search_response": response_data
                    }

        except Exception as e:
            logger.error(f"Error in Exa search: {str(e)}")
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

        search_results = [
            (res["url"], res["text"])
            for res in search_response["search_response"]["results"]
        ]

        return self._format_search_results_for_prompt(search_results)
