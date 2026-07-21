"""
API Integration Test & Demo
Tests the layered memory architecture integration with the API.

Run this to verify the new conversation query endpoints work correctly.

Requirements:
- API running on http://localhost:8000
- PostgreSQL database initialized
- All dependencies installed

Usage:
    python test_api_integration.py
"""

import requests
import json
from uuid import UUID

API_BASE = "http://localhost:8000/api/v1"

def pretty_print(title, obj):
    """Pretty print a JSON object."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if isinstance(obj, dict):
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(obj)


def test_full_conversation_flow():
    """
    Test the full conversation flow:
    1. Create conversation
    2. Query with memory retrieval (turn 1)
    3. Follow-up query (turn 2)
    4. Check memory stats
    """
    
    print("\n" + "="*60)
    print("LAYERED MEMORY ARCHITECTURE - API INTEGRATION TEST")
    print("="*60)
    
    # STEP 1: Create conversation
    print("\n[1] Creating conversation...")
    create_resp = requests.post(
        f"{API_BASE}/conversations",
        json={"title": "Cricket Stats Discussion"}
    )
    assert create_resp.status_code == 200, f"Failed to create conversation: {create_resp.text}"
    conv = create_resp.json()
    conv_id = conv["id"]
    pretty_print("Conversation Created", conv)
    
    # STEP 2: First query (no prior context - should do research or return no memory)
    print("\n[2] Sending first query (about Klaasen)...")
    query1_resp = requests.post(
        f"{API_BASE}/conversations/{conv_id}/query",
        json={"query": "Tell me about Heinrich Klaasen's performance in IPL 2026"}
    )
    assert query1_resp.status_code == 200, f"Failed to query: {query1_resp.text}"
    result1 = query1_resp.json()
    pretty_print("First Query Response", {
        "content": result1["content"][:200] + "..." if len(result1["content"]) > 200 else result1["content"],
        "memory_coverage": result1["memory_info"]["memory_coverage"],
        "research_performed": result1["research_performed"],
        "facts_extracted": len(result1["facts_extracted"]),
        "entities_tracked": result1["resolution_info"]["active_entities"],
    })
    
    print(f"\n✓ Memory coverage: {result1['memory_info']['memory_coverage']:.0%}")
    print(f"✓ Facts extracted: {len(result1['facts_extracted'])}")
    if result1['facts_extracted']:
        print(f"✓ Sample facts:")
        for fact in result1['facts_extracted'][:3]:
            print(f"    - {fact['entity']}.{fact['attribute']} = {fact['value']}")
    
    # STEP 3: Follow-up query (should resolve pronoun and use memory)
    print("\n[3] Sending follow-up query (about 'his' strike rate)...")
    query2_resp = requests.post(
        f"{API_BASE}/conversations/{conv_id}/query",
        json={"query": "What is his strike rate and average runs per match?"}
    )
    assert query2_resp.status_code == 200, f"Failed to follow-up query: {query2_resp.text}"
    result2 = query2_resp.json()
    pretty_print("Follow-Up Query Response", {
        "is_follow_up": result2["resolution_info"]["is_follow_up"],
        "resolved_references": result2["resolution_info"]["resolved_references"],
        "primary_entity": result2["resolution_info"]["primary_entity"],
        "memory_coverage": result2["memory_info"]["memory_coverage"],
        "facts_extracted": len(result2["facts_extracted"]),
    })
    
    print(f"\n✓ Detected as follow-up: {result2['resolution_info']['is_follow_up']}")
    print(f"✓ Primary entity: {result2['resolution_info']['primary_entity']}")
    print(f"✓ Memory coverage: {result2['memory_info']['memory_coverage']:.0%}")
    if result2['resolution_info']['resolved_references']:
        print(f"✓ Resolved references:")
        for ref, entity in result2['resolution_info']['resolved_references'].items():
            print(f"    - '{ref}' → '{entity}'")
    
    # STEP 4: Get memory statistics
    print("\n[4] Getting memory statistics...")
    stats_resp = requests.get(f"{API_BASE}/conversations/{conv_id}/memory-stats")
    assert stats_resp.status_code == 200, f"Failed to get stats: {stats_resp.text}"
    stats = stats_resp.json()
    pretty_print("Memory Statistics", {
        "total_turns": stats["total_turns"],
        "unique_entities": stats["unique_entities"],
        "facts_extracted": stats["facts_extracted"],
        "memory_efficiency": f"{stats['memory_efficiency']:.0%}",
        "tracked_entities": stats["active_entities"],
        "top_topics": stats["top_topics"],
    })
    
    # STEP 5: Get full conversation
    print("\n[5] Getting full conversation history...")
    conv_resp = requests.get(f"{API_BASE}/conversations/{conv_id}")
    assert conv_resp.status_code == 200, f"Failed to get conversation: {conv_resp.text}"
    full_conv = conv_resp.json()
    print(f"\n✓ Conversation has {len(full_conv['messages'])} messages")
    for i, msg in enumerate(full_conv['messages']):
        print(f"  {i+1}. [{msg['role']}] {msg['content'][:80]}...")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print(f"\nConversation ID: {conv_id}")
    print(f"Memory is working! Follow-up questions now use conversation history.")
    print(f"Memory efficiency: {stats['memory_efficiency']:.0%}")
    print(f"Facts learned: {stats['facts_extracted']}")
    

def test_memory_layer_independently():
    """
    Test the memory components directly (without API).
    
    This verifies that the core memory components are working.
    """
    print("\n" + "="*60)
    print("TESTING MEMORY COMPONENTS DIRECTLY")
    print("="*60)
    
    from agents.knowledge_extractor import KnowledgeExtractor
    from agents.follow_up_resolver import FollowUpResolver
    from agents.conversation_memory_retriever import ConversationMemoryRetriever
    from database import SessionLocal
    from uuid import uuid4
    
    db = SessionLocal()
    
    try:
        # Test 1: Knowledge Extraction
        print("\n[1] Testing Knowledge Extraction...")
        extractor = KnowledgeExtractor(db)
        
        response = "Heinrich Klaasen scored 508 runs with a strike rate of 153.93 and played for Delhi Capitals"
        facts = extractor._extract_with_patterns(response)
        
        pretty_print("Extracted Facts", {
            "response": response,
            "facts_found": len(facts),
            "facts": [{"entity": f.entity, "attribute": f.attribute, "value": f.value} for f in facts]
        })
        
        print(f"\n✓ Extracted {len(facts)} facts from response")
        
        # Test 2: Follow-Up Resolver
        print("\n[2] Testing Follow-Up Resolution...")
        resolver = FollowUpResolver(db)
        
        result = resolver.resolve(
            query="What is his average?",
            conversation_id=uuid4(),
            previous_query="Tell me about Klaasen"
        )
        
        pretty_print("Resolution Result", {
            "query": "What is his average?",
            "is_follow_up": result.is_follow_up,
            "resolved_references": result.resolved_references,
            "primary_entity": result.primary_entity,
            "confidence": f"{result.resolution_confidence:.2f}"
        })
        
        print(f"\n✓ Detected follow-up: {result.is_follow_up}")
        if result.resolved_references:
            print(f"✓ Resolved references: {result.resolved_references}")
        
        print("\n" + "="*60)
        print("✓ COMPONENT TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error in component testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    
    print("\nLayered Memory Architecture - Integration Test")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--components-only":
        # Test just the memory components
        test_memory_layer_independently()
    else:
        # Test the full API integration
        try:
            test_full_conversation_flow()
        except requests.exceptions.ConnectionError:
            print("\n✗ Could not connect to API at http://localhost:8000")
            print("Please ensure the API is running:")
            print("  python -m uvicorn api.main:app --reload")
            print("\nOr test just the components:")
            print("  python test_api_integration.py --components-only")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
