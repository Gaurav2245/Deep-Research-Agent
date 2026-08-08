from agents.graph import _route_after_query_planner, _should_continue_research
from agents.state import ResearchState


def make_state(**overrides) -> ResearchState:
    state = ResearchState(query="What is the capital of France?")
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_should_continue_research_stops_when_no_pending_queries():
    state = make_state(search_queries=[], should_continue_research=True)
    assert _should_continue_research(state) == "synthesise"


def test_should_continue_research_continues_when_confidence_low():
    state = make_state(search_queries=["q2"], should_continue_research=True)
    assert _should_continue_research(state) == "web_search"


def test_should_continue_research_stops_when_confidence_sufficient():
    state = make_state(search_queries=["q2"], should_continue_research=False)
    assert _should_continue_research(state) == "synthesise"


def test_route_after_query_planner_uses_new_queries_if_present():
    state = make_state(search_queries=["new query"])
    assert _route_after_query_planner(state) == "web_search"


def test_route_after_query_planner_synthesises_with_prior_sources():
    state = make_state(search_queries=[], scored_sources=[{"url": "https://x.com"}])
    assert _route_after_query_planner(state) == "synthesise"


def test_route_after_query_planner_synthesises_with_prior_chat_history():
    state = make_state(
        search_queries=[],
        chat_history=[
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
    )
    assert _route_after_query_planner(state) == "synthesise"


def test_route_after_query_planner_falls_back_to_raw_query_when_no_context():
    state = make_state(search_queries=[], chat_history=[])
    result = _route_after_query_planner(state)
    assert result == "web_search"
    assert state.search_queries == [state.query]
    assert state.query in state.attempted_queries
