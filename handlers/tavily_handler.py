import aiohttp
import os
import logging

from typing import Dict, Any, Optional
from dotenv import load_dotenv

from handlers.base_handler import ProviderHandler

load_dotenv()

logger = logging.getLogger(__name__)

TAVILY_API_URL = 'https://api.tavily.com'


class TavilyHandler(ProviderHandler):
    """Handles interactions with the Tavily API."""

    def __init__(
            self,
            search_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize the TavilyHandler.

        Args:
            search_params: Default search parameters to use for all searches
        """
        super().__init__(
            api_key=os.getenv("TAVILY_API_KEY"),
            api_url=TAVILY_API_URL,
            search_params=search_params
        )
        self.is_llm_response = search_params.get("include_answer", False)

    async def search(self, query: str) -> Dict[str, Any]:
        """Run a Tavily search using async HTTP request.

        Args:
            query: The query to search for

        Returns:
            Dictionary containing 'answer' and 'raw_response'
        """
        headers = {
            'Content-Type': 'application/json',
        }

        # Construct request data
        data = {
            'query': query,
            'api_key': self.api_key,
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
                        print(f"Error in Tavily search: HTTP {response.status}")
                        return {
                            "answer": "",
                            "raw_response": None
                        }

                    response_data = await response.json()
                    answer = response_data.get("answer", "") if self.is_llm_response else ""
                    return {
                        "answer": answer,
                        "search_response": response_data
                    }

        except Exception as e:
            print(f"Error in Tavily search: {str(e)}")
            return {
                "answer": "",
                "raw_response": None
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
            (res['url'], res['content'])
            for res in search_response['search_response']['results']
        ]

        return self._format_search_results_for_prompt(search_results)
