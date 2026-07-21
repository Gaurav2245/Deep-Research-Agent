"""
config/settings.py
Centralised configuration loaded from environment / .env file.
All other modules import from here — never read os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


# ── Enums ────────────────────────────────────────────────────────────────────

class SearchProvider(str, Enum):
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    NSE = "nse"
    PLAYWRIGHT = "playwright"


class LLMProvider(str, Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"


class ResearchDepth(str, Enum):
    SHALLOW = "shallow"
    DEEP = "deep"


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AzureOpenAIConfig:
    api_key: str = field(default_factory=lambda: os.environ["AZURE_OPENAI_API_KEY"])
    endpoint: str = field(default_factory=lambda: os.environ["AZURE_OPENAI_ENDPOINT"])
    deployment_name: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    )
    embedding_deployment_name: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")
    )
    api_version: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    )


@dataclass(frozen=True)
class TavilyConfig:
    api_key: str = field(default_factory=lambda: os.environ["TAVILY_API_KEY"])
    max_results: int = field(
        default_factory=lambda: int(os.getenv("MAX_RESULTS_PER_SEARCH", "5"))
    )


@dataclass(frozen=True)
class DuckDuckGoConfig:
    max_results: int = field(
        default_factory=lambda: int(os.getenv("MAX_RESULTS_PER_SEARCH", "5"))
    )


@dataclass(frozen=True)
class ScraperConfig:
    headless: bool = field(
        default_factory=lambda: os.getenv("SCRAPER_HEADLESS", "true").lower() == "true"
    )
    timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("SCRAPER_TIMEOUT_MS", "30000"))
    )
    max_scrape_urls: int = field(
        default_factory=lambda: int(os.getenv("SCRAPER_MAX_URLS", "3"))
    )


@dataclass(frozen=True)
class NSEConfig:
    timeout: int = field(
        default_factory=lambda: int(os.getenv("NSE_TIMEOUT", "20"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("NSE_MAX_RETRIES", "3"))
    )
    headless: bool = field(
        default_factory=lambda: os.getenv("NSE_HEADLESS", "true").lower() == "true"
    )


@dataclass(frozen=True)
class AgentConfig:
    search_provider: SearchProvider = field(
        default_factory=lambda: SearchProvider(
            os.getenv("SEARCH_PROVIDER", "tavily").split("#")[0].strip().lower()
        )
    )
    llm_provider: LLMProvider = field(
        default_factory=lambda: LLMProvider(
            os.getenv("LLM_PROVIDER", "azure_openai").split("#")[0].strip().lower()
        )
    )
    max_search_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_ITERATIONS", "3"))
    )
    research_depth: ResearchDepth = field(
        default_factory=lambda: ResearchDepth(
            os.getenv("RESEARCH_DEPTH", "deep").split("#")[0].strip().lower()
        )
    )
    enable_follow_up_searches: bool = field(
        default_factory=lambda: os.getenv(
            "ENABLE_FOLLOW_UP_SEARCHES", "true"
        ).lower() == "true"
    )
    enable_scraper: bool = field(
        default_factory=lambda: os.getenv(
            "ENABLE_SCRAPER", "true"
        ).lower() == "true"
    )
    # LangSmith observability
    langsmith_tracing: bool = field(
        default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )
    langsmith_api_key: str | None = field(
        default_factory=lambda: os.getenv("LANGCHAIN_API_KEY")
    )
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "deep-research-agent")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )


# ── Singleton accessors ───────────────────────────────────────────────────────

def get_agent_config() -> AgentConfig:
    return AgentConfig()


def get_azure_openai_config() -> AzureOpenAIConfig:
    return AzureOpenAIConfig()


def get_tavily_config() -> TavilyConfig:
    return TavilyConfig()


def get_duckduckgo_config() -> DuckDuckGoConfig:
    return DuckDuckGoConfig()


def get_scraper_config() -> ScraperConfig:
    return ScraperConfig()


def get_nse_config() -> NSEConfig:
    return NSEConfig()
