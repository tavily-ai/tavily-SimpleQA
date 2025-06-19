import logging
from typing import Dict, Any, Optional

from handlers.base_handler import ProviderHandler
from gpt_researcher import GPTResearcher

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "utils/handlers/configs/gptr_config.json"


class GPTRHandler(ProviderHandler):
    """Handles research using the GPT Researcher package."""

    def __init__(
            self,
            search_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize the GPTRHandler.

        Args:
            search_params: Default search parameters to use for all searches
        """
        # GPTResearcher will read API keys from environment variables
        # We just need a dummy API key for the base handler
        super().__init__(
            api_key="dummy_key",
            api_url="dummy_url",
            search_params=search_params
        )
        self.is_llm_response = True

    async def search(self, query: str) -> Dict[str, Any]:
        """Run research using GPT Researcher.

        Args:
            query: The query to search for

        Returns:
            Dictionary containing 'answer' and 'raw_response'
        """
        try:
            researcher = GPTResearcher(
                query=query,
                report_type=self.search_params.get("report_type", "deep"),
                config_path=self.search_params.get("config_path", DEFAULT_CONFIG_PATH)
            )

            logger.info(f"GPT Researcher starting research for: {query}")
            research_result = await researcher.conduct_research()

            logger.info(f"GPT Researcher completed research for: {query}")
            return {
                "answer": research_result,
                "search_response": research_result
            }

        except Exception as e:
            logger.error(f"Error in GPT Researcher: {str(e)}")
            return {
                "answer": "",
                "search_response": None
            }

    async def post_process(self, search_response: str) -> str:
        """Do nothing for GPT Researcher - answer is already in the search response"""
        return search_response
