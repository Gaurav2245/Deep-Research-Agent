
from agents import build_research_graph, ResearchState
from config import get_agent_config
from llm.factory import create_llm

cfg = get_agent_config()
graph = build_research_graph(cfg)
initial_state = ResearchState(query="test")
final_state = graph.invoke(initial_state)

print(f"Type of final_state: {type(final_state)}")
if isinstance(final_state, dict):
    print("It is a dict")
else:
    print("It is not a dict")
