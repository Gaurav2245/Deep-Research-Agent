from langgraph.graph import StateGraph, START, END
from agents.enhanced_nodes import (
    make_confidence_scorer_node,
    make_database_persistence_node,
    make_data_validator_node,
    make_embedder_node,
    make_source_scorer_node,
)
from agents.query_understanding import make_query_understanding_node
from agents.nodes import (
    make_entity_extractor_node,
    make_follow_up_node,
    make_query_planner_node,
    make_relational_knowledge_extractor_node,
    make_synthesiser_node,
    make_web_search_node,
)
from agents.scraper_node import make_scraper_node
from agents.conversational_knowledge import prior_context_available_for_answer
from agents.state import ResearchState
from config import AgentConfig, get_agent_config
from llm.factory import create_llm
from tools.factory import create_search_tool
from tools.playwright_scraper import PlaywrightScraperTool, ScraperConfig
from utils.logger import get_logger

logger = get_logger(__name__)


# Router logic

def _should_continue_research(state: ResearchState) -> str:
    cfg = get_agent_config()

    # Dead-end: graph only goes to web_search — no queries means zero progress forever
    pending = getattr(state, "search_queries", None) or []
    if not pending:
        logger.info(
            "[Router] No pending search queries; forcing synthesise (exhausted / no work)."
        )
        return "synthesise"

    # Use confidence-based stop condition if available
    if hasattr(state, 'should_continue_research'):
        if not state.should_continue_research:
            logger.info("[Router] Confidence sufficient or max iterations reached. Reason: %s", state.confidence_reason)
            return "synthesise"
        else:
            logger.info("[Router] Confidence low, continuing research. Reason: %s", state.confidence_reason)
            return "web_search"
            
    # Fallback to simple iteration count
    if state.search_queries and state.iteration < cfg.max_search_iterations:
        logger.debug(
            "[Router] More research needed | iteration=%d/%d | queries=%d",
            state.iteration, cfg.max_search_iterations, len(state.search_queries),
        )
        return "web_search"
    logger.debug("[Router] Research complete")
    return "synthesise"


def _route_after_query_planner(state: ResearchState) -> str:
    """
    If the planner chose no new searches and we already have prior evidence or
    chat answers, skip web_search / scoring / follow-up loop and synthesise once.
    """
    if state.search_queries:
        return "web_search"
    prior = prior_context_available_for_answer(
        scored_sources=getattr(state, "scored_sources", None),
        context=getattr(state, "context", None),
        conversational_knowledge=getattr(state, "conversational_knowledge", None),
        chat_history=getattr(state, "chat_history", None),
    )
    if prior:
        logger.info(
            "[Router] No new search queries; prior sources/context/memory available → synthesise."
        )
        return "synthesise"
    h = getattr(state, "chat_history", None) or []
    if len(h) <= 1:
        logger.warning(
            "[Router] No search queries and no prior context; using raw query for web search."
        )
        state.search_queries = [state.query]
        if state.query and state.query not in state.attempted_queries:
            state.attempted_queries.append(state.query)
        return "web_search"
    logger.info("[Router] No new queries; answering from conversation text only → synthesise.")
    return "synthesise"


# Graph builder

def build_research_graph(config: AgentConfig | None = None):
    """
    Build and compile the LangGraph research workflow.
    """
    cfg = config or get_agent_config()
    llm = create_llm(cfg)
    search_tool = create_search_tool(cfg)

    # Nodes
    query_understanding = make_query_understanding_node(llm)
    entity_extractor = make_entity_extractor_node(llm)
    query_planner = make_query_planner_node(llm, cfg)
    web_search    = make_web_search_node(search_tool, cfg)
    follow_up     = make_follow_up_node(llm, cfg)
    synthesiser   = make_synthesiser_node(llm)
    relational_memory = make_relational_knowledge_extractor_node(llm)
    
    # Enhanced nodes
    source_scorer = make_source_scorer_node(cfg)
    embedder = make_embedder_node(cfg)
    data_validator = make_data_validator_node(cfg)
    confidence_scorer = make_confidence_scorer_node(cfg)
    db_persistence = make_database_persistence_node()

    # Graph
    builder: StateGraph = StateGraph(ResearchState)

    builder.add_node("query_understanding", query_understanding)
    builder.add_node("entity_extractor", entity_extractor)
    builder.add_node("query_planner", query_planner)
    builder.add_node("web_search", web_search)
    builder.add_node("source_scorer", source_scorer)
    builder.add_node("embedder", embedder)
    builder.add_node("follow_up", follow_up)
    builder.add_node("confidence_scorer", confidence_scorer)
    builder.add_node("synthesise", synthesiser)
    builder.add_node("relational_memory", relational_memory)
    builder.add_node("data_validator", data_validator)
    builder.add_node("db_persistence", db_persistence)

    # Edges
    builder.add_edge(START, "query_understanding")
    builder.add_edge("query_understanding", "entity_extractor")
    builder.add_edge("entity_extractor", "query_planner")
    builder.add_conditional_edges(
        "query_planner",
        _route_after_query_planner,
        {"web_search": "web_search", "synthesise": "synthesise"},
    )

    builder.add_edge("web_search", "source_scorer")
    builder.add_edge("source_scorer", "embedder")
    
    if cfg.enable_scraper:
        from config import get_scraper_config
        scr_cfg = get_scraper_config()
        # Use the ScraperConfig from tools.playwright_scraper but with values from config.settings
        playwright_cfg = ScraperConfig(
            headless=scr_cfg.headless,
            timeout_ms=scr_cfg.timeout_ms
        )
        scraper = PlaywrightScraperTool(config=playwright_cfg)
        scraper_node = make_scraper_node(
            scraper=scraper, 
            max_urls=scr_cfg.max_scrape_urls
        )
        builder.add_node("scraper_actual", scraper_node)
        builder.add_edge("embedder", "scraper_actual")
        builder.add_edge("scraper_actual", "follow_up")
    else:
        builder.add_edge("embedder", "follow_up")

    builder.add_edge("follow_up", "confidence_scorer")

    builder.add_conditional_edges(
        "confidence_scorer",
        _should_continue_research,
        {"web_search": "web_search", "synthesise": "synthesise"},
    )
    
    builder.add_edge("synthesise", "relational_memory")
    builder.add_edge("relational_memory", "data_validator")
    builder.add_edge("data_validator", "db_persistence")
    builder.add_edge("db_persistence", END)

    graph = builder.compile()
    logger.info(
        "Research graph compiled | search=%s llm=%s",
        search_tool.provider_name(),
        cfg.llm_provider.value,
    )
    return graph
