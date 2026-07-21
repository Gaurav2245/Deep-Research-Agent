"""
main.py
CLI entry point for the Deep Research Agent.

Usage
-----
    python main.py "What are the latest developments in quantum computing?"

Or import and use programmatically:
    from main import run_research
    result = run_research("Your question here")
    print(result.final_answer)
"""
import os
import sys

from agents import ResearchState, build_research_graph
from agents.state import ResearchState as StateClass
from config import get_agent_config
from database.connection import init_db
from utils.logger import get_logger

logger = get_logger(__name__)


def setup_langsmith():
    """Setup LangSmith tracing if enabled in config."""
    cfg = get_agent_config()
    if cfg.langsmith_tracing:
        logger.info("LangSmith tracing enabled")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if cfg.langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith_project


def run_research(query: str) -> StateClass:
    """
    Run the full research pipeline for a given query.
    """
    cfg = get_agent_config()
    logger.setLevel(cfg.log_level)
    
    setup_langsmith()
    
    # Initialize DB (create tables if they don't exist)
    try:
        init_db()
    except Exception as e:
        logger.warning("Database initialization failed (is Postgres running?): %s", e)

    logger.info("Deep Research Agent starting")
    logger.info("Query       : %s", query)
    logger.info("Search      : %s", cfg.search_provider.value)
    logger.info("LLM         : %s", cfg.llm_provider.value)

    graph = build_research_graph(cfg)
    initial_state = StateClass(query=query)

    final_state = graph.invoke(initial_state)
    
    # If LangGraph returns a dict (depending on config/version), convert back to ResearchState
    if isinstance(final_state, dict):
        logger.debug("Graph returned dict, converting to ResearchState")
        # Use constructor to pick up all fields from dict
        final_state = StateClass(**{k: v for k, v in final_state.items() if k in StateClass.__dataclass_fields__})
    
    logger.info("Research complete | iterations=%d | confidence=%.2f", 
                final_state.iteration, final_state.confidence_score)
    return final_state


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<your research query>\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = run_research(query)

    print("\nRESEARCH ANSWER")
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(result.final_answer or "No answer generated.")

    print(f"\nSources searched : {len(result.search_responses)} rounds")
    print(f"Queries used     : {result.search_queries}")


if __name__ == "__main__":
    main()
