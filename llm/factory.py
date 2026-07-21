"""
llm/factory.py
Returns the configured LLM.  Add new providers here only.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from config import AgentConfig, LLMProvider, get_agent_config
from utils.logger import get_logger

logger = get_logger(__name__)


def create_llm(
    config: AgentConfig | None = None,
    temperature: float = 0.1,
    streaming: bool = True,
) -> BaseChatModel:
    """
    Factory that returns the correct LLM based on LLM_PROVIDER env var.

    Raises
    ------
    ValueError
        If the configured LLM_PROVIDER is not supported.
    """
    cfg = config or get_agent_config()

    if cfg.llm_provider == LLMProvider.AZURE_OPENAI:
        from llm.azure_openai import create_azure_llm
        return create_azure_llm(temperature=temperature, streaming=streaming)

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{cfg.llm_provider.value}'. "
        f"Valid choices: {[p.value for p in LLMProvider]}"
    )
