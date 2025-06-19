import os
import logging
import aiohttp
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from handlers.base_handler import ProviderHandler

load_dotenv()

logger = logging.getLogger(__name__)

PERPLEXITY_API_URL = "https://api.perplexity.ai"
DEFAULT_MODEL = "sonar-pro"


class PerplexityHandler(ProviderHandler):
    """Handles interactions with the Perplexity API."""

    def __init__(
            self,
            search_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the PerplexityHandler.

        Args:
            search_params: Default search parameters to use for all searches
        """
        super().__init__(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            api_url=PERPLEXITY_API_URL,
            search_params=search_params
        )
        self.model = search_params.get("model", DEFAULT_MODEL)
        self.is_llm_response = True

    async def search(self, query: str) -> Dict[str, Any]:
        """Run a Perplexity search using async HTTP request.

        Args:
            query: The query to search for

        Returns:
            Dictionary containing 'answer' and 'search_response'
        """
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }

        messages = [
            {
                "role": "system",
                "content": "Be precise and concise."
            },
            {
                "role": "user",
                "content": query
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.api_url}/chat/completions",
                        json=payload,
                        headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Error in Perplexity search: HTTP {response.status}")
                        error_text = await response.text()
                        logger.error(f"Response: {error_text}")
                        return {
                            "answer": "",
                            "search_response": None
                        }

                    response_data = await response.json()
                    logger.info("Received response from Perplexity API")

                    sources = self._extract_sources(response_data)
                    answer = self._construct_answer(response_data, sources)

                    return {
                        "answer": answer,
                        "search_response": response_data
                    }

        except Exception as e:
            logger.error(f"Error in Perplexity search: {str(e)}")
            return {
                "answer": "",
                "search_response": None
            }

    def _extract_sources(self, response_data: Dict[str, Any]) -> str:
        """
        Extract and format citation sources from Perplexity response data.

        Args:
            response_data: The raw response data from Perplexity API

        Returns:
            Formatted citations as a numbered list
        """
        formatted_citations = ""

        if "citations" in response_data and isinstance(response_data["citations"], list):
            for i, citation in enumerate(response_data["citations"], 1):
                formatted_citations += f"[{i}] {citation}\n"

            return formatted_citations

        return formatted_citations

    def _construct_answer(self, response_data: Dict[str, Any], sources: str) -> str:
        """
        Construct the answer from the Perplexity response data and sources.
        """
        choices = response_data.get("choices", [{}])
        answer = ""
        for choice in choices:
            msg = choice.get("message", {}).get("content", "")
            logger.info(f"Adding to answer: {msg}")
            if msg:
                answer += msg

        if sources:
            answer += f"\nSources:\n{sources}"

        return answer

    async def post_process(self, search_response: str) -> str:
        """Do nothing for Perplexity Sonar - answer is already in the search response"""
        return search_response
