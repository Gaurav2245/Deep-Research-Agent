
from __future__ import annotations

from config import AgentConfig, SearchProvider, get_agent_config
from tools.base import BaseSearchTool
from utils.logger import get_logger

logger = get_logger(__name__)


def create_search_tool(config: AgentConfig | None = None) -> BaseSearchTool:
    """
    Instantiate and return the configured search tool.

    Parameters
    ----------
    config:
        AgentConfig instance. If None, the default (env-based) config is used.

    Returns
    -------
    BaseSearchTool
        A fully initialised, provider-specific search tool.

    Raises
    ------
    ValueError
        If the configured SEARCH_PROVIDER is not supported.
    """
    cfg = config or get_agent_config()
    provider = cfg.search_provider

    logger.info("Creating search tool | provider=%s", provider.value)

    if provider == SearchProvider.TAVILY:
        from tools.tavily_tool import TavilySearchTool
        from tools.fallback_tool import FallbackSearchTool

        return FallbackSearchTool(
        primary=TavilySearchTool()
    )

    if provider == SearchProvider.DUCKDUCKGO:
        from tools.duckduckgo_tool import DuckDuckGoSearchTool
        return DuckDuckGoSearchTool()

    if provider == SearchProvider.NSE:
        from tools.nse_tool import NSETool
        return NSETool()

    if provider == SearchProvider.PLAYWRIGHT:
        from tools.playwright_scraper import PlaywrightScraperTool
        return PlaywrightScraperTool()

    raise ValueError(
        f"Unsupported SEARCH_PROVIDER '{provider.value}'. "
        f"Valid choices: {[p.value for p in SearchProvider]}"
    )
