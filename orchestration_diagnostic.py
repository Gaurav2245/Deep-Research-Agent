"""
Orchestration Diagnostic

Checks which nodes are already in graph vs. which are missing.
Shows the integration checklist status.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.graph import build_research_graph
from agents.state import ResearchState
from config import get_agent_config
from utils.logger import get_logger

logger = get_logger(__name__)


def diagnose_orchestration():
    """Check graph structure against orchestration requirements"""
    
    print("\n" + "="*70)
    print("ORCHESTRATION DIAGNOSTIC")
    print("="*70)
    
    try:
        # Build graph
        graph = build_research_graph()
        
        # Get all nodes
        nodes = graph.nodes if hasattr(graph, 'nodes') else {}
        node_names = list(nodes.keys()) if nodes else []
        
        print(f"\n✓ Graph built successfully")
        print(f"  Nodes in graph: {len(node_names)}")
        
        # Check required nodes
        required_nodes = {
            "conversation_state_reconstructor": "Load prior context from memory",
            "conversation_memory_retriever": "Query conversation facts before web search",
            "evidence_gating": "Block synthesis if evidence insufficient",
            "claim_level_grounding": "Validate grounding at claim level",
        }
        
        print("\n" + "-"*70)
        print("REQUIRED NODES:")
        print("-"*70)
        
        missing_nodes = []
        present_nodes = []
        
        for node_name, purpose in required_nodes.items():
            if node_name in node_names:
                print(f"✓ {node_name:40} {purpose}")
                present_nodes.append(node_name)
            else:
                print(f"✗ {node_name:40} MISSING!")
                missing_nodes.append(node_name)
        
        # Check existing nodes
        print("\n" + "-"*70)
        print("EXISTING NODES:")
        print("-"*70)
        
        existing = [n for n in node_names if n not in required_nodes]
        for node in existing:
            print(f"  • {node}")
        
        # Integration status
        print("\n" + "-"*70)
        print("INTEGRATION STATUS:")
        print("-"*70)
        
        integration_status = {
            "State Reconstruction": (
                "conversation_state_reconstructor" in node_names,
                "Loads prior context before planning"
            ),
            "Memory Retrieval": (
                "conversation_memory_retriever" in node_names,
                "Queries memory before web search"
            ),
            "Evidence Gating": (
                "evidence_gating" in node_names,
                "Blocks synthesis without evidence"
            ),
            "Claim Validation": (
                "claim_level_grounding" in node_names,
                "Validates claims granularly"
            ),
        }
        
        for feature, (present, description) in integration_status.items():
            status = "✓" if present else "✗"
            print(f"{status} {feature:25} {description}")
        
        # Summary
        print("\n" + "-"*70)
        print("SUMMARY:")
        print("-"*70)
        
        total_required = len(required_nodes)
        total_present = len(present_nodes)
        integration_pct = (total_present / total_required) * 100 if total_required > 0 else 0
        
        print(f"Integration: {total_present}/{total_required} nodes ({integration_pct:.0f}%)")
        
        if missing_nodes:
            print(f"\nMissing nodes (need to add to graph):")
            for node in missing_nodes:
                print(f"  1. {node}")
            print("\nNext steps:")
            print("  1. Add ConversationStateReconstructor node after entity_extractor")
            print("  2. Add ConversationMemoryRetriever node after state reconstruction")
            print("  3. Add EvidenceGating node after confidence_scorer")
            print("  4. Add ClaimLevelGrounding node after synthesiser")
        else:
            print("\n✓ All required nodes present!")
        
        # Check state fields
        print("\n" + "-"*70)
        print("STATE FIELD SUPPORT:")
        print("-"*70)
        
        state_fields = [
            "conversation_id",
            "conversation_state_context",
            "memory_retrieval_result",
            "evidence_profile",
            "claim_grounding_result",
            "synthesis_constraints",
            "synthesis_blocked",
        ]
        
        # Get ResearchState fields
        from dataclasses import fields as dataclass_fields
        state_field_names = [f.name for f in dataclass_fields(ResearchState)]
        
        for field in state_fields:
            if field in state_field_names:
                print(f"✓ {field}")
            else:
                print(f"? {field} (may be added dynamically)")
        
        print("\n" + "="*70)
        print("END DIAGNOSTIC")
        print("="*70 + "\n")
        
        return {
            "total_nodes": len(node_names),
            "required_nodes_present": total_present,
            "required_nodes_total": total_required,
            "integration_pct": integration_pct,
            "missing_nodes": missing_nodes,
        }
        
    except Exception as e:
        logger.error(f"Error in diagnostic: {e}", exc_info=True)
        print(f"\n✗ Error during diagnostic: {e}")
        return None


def check_data_flow():
    """Check if data flows through orchestration correctly"""
    
    print("\n" + "="*70)
    print("DATA FLOW DIAGNOSTIC")
    print("="*70)
    
    print("\nExpected data flow:")
    print("1. User query")
    print("   ↓")
    print("2. State Reconstruction")
    print("   Prior entities: [loaded from memory]")
    print("   Conversation topics: [inferred]")
    print("   ↓")
    print("3. Memory Retrieval")
    print("   Retrieved entities: [from prior answers]")
    print("   Retrieved claims: [facts from memory]")
    print("   ↓")
    print("4. Query Planner")
    print("   Context injection: [conversation + memory]")
    print("   ↓")
    print("5. Web Search")
    print("   Searches for: [targeted, context-aware queries]")
    print("   ↓")
    print("6. Source Scoring")
    print("   Scores: [15 sources]")
    print("   ↓")
    print("7. Evidence Gating")
    print("   Check: [sources sufficient? YES/NO]")
    print("   ↓")
    print("8. Synthesis")
    print("   Generate: [with constraints, only grounded entities]")
    print("   ↓")
    print("9. Claim-Level Grounding")
    print("   Validate: [each claim individually]")
    print("   ↓")
    print("10. Store")
    print("    Save: [conversation state, entities, trace]")
    print("    ↓")
    print("11. Response to User")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Run diagnostics
    result = diagnose_orchestration()
    check_data_flow()
    
    if result and result['integration_pct'] == 100:
        print("✓ Orchestration fully integrated!")
        sys.exit(0)
    elif result and result['integration_pct'] > 50:
        print("⚠ Orchestration partially integrated - missing steps above")
        sys.exit(1)
    else:
        print("✗ Orchestration not yet integrated - follow the integration guide")
        sys.exit(1)
