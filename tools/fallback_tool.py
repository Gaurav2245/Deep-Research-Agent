from tools.base import BaseSearchTool, SearchResponse
from tools.duckduckgo_tool import DuckDuckGoSearchTool
from utils.logger import get_logger

logger = get_logger(__name__)


class FallbackSearchTool(BaseSearchTool):
    def __init__(self, primary: BaseSearchTool):
        self.primary = primary
        self.fallback = DuckDuckGoSearchTool()   # ✅ fallback set here

    def provider_name(self) -> str:
        return f"{self.primary.provider_name()} → DuckDuckGo"

    def search(self, query: str, **kwargs) -> SearchResponse:
        try:
            logger.info("Trying primary search: %s", self.primary.provider_name())
            result = self.primary.search(query, **kwargs)

            if result and result.results:
                return result

            logger.warning("Primary returned empty results → switching to DuckDuckGo")

        except Exception as e:
            logger.warning("Primary search failed: %s → switching to DuckDuckGo", e)

        # 🔁 fallback always DuckDuckGo
        logger.info("Using fallback search: DuckDuckGo")
        return self.fallback.search(query, **kwargs)