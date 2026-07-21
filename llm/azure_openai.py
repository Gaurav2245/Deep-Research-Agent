
from __future__ import annotations

from langchain_openai import AzureChatOpenAI

from config import AzureOpenAIConfig, get_azure_openai_config
from utils.logger import get_logger

logger = get_logger(__name__)


def create_azure_llm(
    config: AzureOpenAIConfig | None = None,
    temperature: float = 0.1,
    streaming: bool = True,
) -> AzureChatOpenAI:
    """
    Build and return an AzureChatOpenAI instance.

    Parameters
    ----------
    config:
        Explicit config. Falls back to env-based config when None.
    temperature:
        Sampling temperature (0 = deterministic, useful for research).
    streaming:
        Whether to enable token streaming.
    """
    cfg = config or get_azure_openai_config()

    logger.info(
        "Creating Azure OpenAI LLM | deployment=%s endpoint=%s",
        cfg.deployment_name,
        cfg.endpoint,
    )

    return AzureChatOpenAI(
        azure_endpoint=cfg.endpoint,
        azure_deployment=cfg.deployment_name,
        openai_api_key=cfg.api_key,
        openai_api_version=cfg.api_version,
        temperature=temperature,
        streaming=streaming,
    )
