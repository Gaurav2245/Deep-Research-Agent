
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

        tool: BaseSearchTool = FallbackSearchTool(primary=TavilySearchTool())
        return _maybe_wrap_with_nse_routing(tool, cfg)

    if provider == SearchProvider.DUCKDUCKGO:
        from tools.duckduckgo_tool import DuckDuckGoSearchTool
        return _maybe_wrap_with_nse_routing(DuckDuckGoSearchTool(), cfg)

    if provider == SearchProvider.NSE:
        from config import get_nse_config
        from tools.nse_tool import NSEConfig, NSETool

        nse_cfg = get_nse_config()
        return NSETool(
            NSEConfig(
                timeout=nse_cfg.timeout,
                max_retries=nse_cfg.max_retries,
                headless=nse_cfg.headless,
            )
        )

    if provider == SearchProvider.PLAYWRIGHT:
        from tools.playwright_scraper import PlaywrightScraperTool
        return PlaywrightScraperTool()

    raise ValueError(
        f"Unsupported SEARCH_PROVIDER '{provider.value}'. "
        f"Valid choices: {[p.value for p in SearchProvider]}"
    )


def _maybe_wrap_with_nse_routing(tool: BaseSearchTool, cfg: AgentConfig) -> BaseSearchTool:
    """
    Wrap a web-search tool so Indian stock-market queries (nifty/sensex/gainers/
    losers) are routed to NSETool's live data instead of generic web search.
    Controlled by ENABLE_NSE_AUTO_ROUTING (default: on).
    """
    if not cfg.enable_nse_auto_routing:
        return tool
    from tools.nse_router_tool import AutoRoutingSearchTool
    return AutoRoutingSearchTool(default_tool=tool)
