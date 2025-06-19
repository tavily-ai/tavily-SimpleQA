from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class ProviderHandler(ABC):
    """Abstract base class for handling interactions with a search API provider."""

    def __init__(
            self,
            api_key: Optional[str] = None,
            api_url: str = None,
            search_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize the ProviderHandler.

        Args:
            api_key: API key for the search provider (defaults to env variable if not passed)
            api_url: Base URL for the search API
            search_params: Default search parameters to use for all searches
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "API key not provided to initialize search provider handler"
            )

        self.api_url = api_url
        if not self.api_url:
            raise ValueError(
                "API url not provided to initialize search provider handler"
            )

        # Store default search parameters
        self.search_params = search_params or {}

    @abstractmethod
    async def search(self, query: str) -> Dict[str, Any]:
        """Run a search using async HTTP request.

        Args:
            query: The query to search for

        Returns:
            Dictionary containing 'answer' and 'raw_response'
        """
        pass

    @abstractmethod
    async def post_process(self, search_response: dict) -> list:
        """
        Post process search response.

        Args:
            search_response: Dictionary containing the search response

        Returns:
            processed response ready for LLM prompt
        """
        pass

    def _format_search_results_for_prompt(self, search_results: list) -> str:
        """
        Private helper to format search results into a string with document numbers, URLs, and content.
        Args:
            search_results: List of (url, content) tuples
        Returns:
            str: Formatted string
        """
        return "\n".join(
            f"\n**Document {i + 1}.** Source: {url}\nContent: {content}"
            for i, (url, content) in enumerate(search_results)
        )
