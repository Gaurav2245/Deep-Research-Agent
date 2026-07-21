#!/usr/bin/env python3
"""
QUICK START - Layered Memory Architecture API

Copy-paste examples to test the new conversation endpoints.

Usage:
    python quick_start.py
"""

import requests
import json

API = "http://localhost:8000/api/v1"

print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   LAYERED CONVERSATIONAL MEMORY ARCHITECTURE - QUICK START     ║
║                                                                ║
║   New Endpoints:                                               ║
║   • POST /conversations/{id}/query                            ║
║   • GET /conversations/{id}/memory-stats                      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")

# Test 1: Create conversation
print("\n[1] Creating conversation...")
try:
    r = requests.post(f"{API}/conversations", json={"title": "Cricket Stats"})
    r.raise_for_status()
    conv = r.json()
    conv_id = conv["id"]
    print(f"✓ Created conversation: {conv_id}")
except Exception as e:
    print(f"✗ Error: {e}")
    print("  Make sure API is running: python -m uvicorn api.main:app --reload")
    exit(1)

# Test 2: First query
print("\n[2] Sending first query...")
try:
    query = "Tell me about Virat Kohli's IPL 2026 season"
    print(f"  Query: {query}")
    
    r = requests.post(
        f"{API}/conversations/{conv_id}/query",
        json={"query": query}
    )
    r.raise_for_status()
    result = r.json()
    
    print(f"✓ Response received")
    print(f"  - Memory coverage: {result['memory_info']['memory_coverage']:.0%}")
    print(f"  - Was follow-up: {result['resolution_info']['is_follow_up']}")
    print(f"  - Facts extracted: {len(result['facts_extracted'])}")
    print(f"  - Response time: {result['elapsed_ms']:.0f}ms")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 3: Follow-up query (test pronoun resolution)
print("\n[3] Sending follow-up query...")
try:
    query = "What is his batting average this season?"
    print(f"  Query: {query}")
    
    r = requests.post(
        f"{API}/conversations/{conv_id}/query",
        json={"query": query}
    )
    r.raise_for_status()
    result = r.json()
    
    print(f"✓ Response received")
    print(f"  - Was follow-up: {result['resolution_info']['is_follow_up']}")
    print(f"  - Primary entity: {result['resolution_info']['primary_entity']}")
    if result['resolution_info']['resolved_references']:
        for ref, entity in result['resolution_info']['resolved_references'].items():
            print(f"  - Resolved '{ref}' → '{entity}'")
    print(f"  - Memory coverage: {result['memory_info']['memory_coverage']:.0%}")
    print(f"  - Facts extracted: {len(result['facts_extracted'])}")
    if result['facts_extracted']:
        for fact in result['facts_extracted'][:2]:
            print(f"    • {fact['entity']}.{fact['attribute']} = {fact['value']}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 4: Get memory stats
print("\n[4] Getting memory statistics...")
try:
    r = requests.get(f"{API}/conversations/{conv_id}/memory-stats")
    r.raise_for_status()
    stats = r.json()
    
    print(f"✓ Statistics retrieved")
    print(f"  - Total turns: {stats['total_turns']}")
    print(f"  - Unique entities: {stats['unique_entities']}")
    print(f"  - Facts learned: {stats['facts_extracted']}")
    print(f"  - Memory efficiency: {stats['memory_efficiency']:.0%}")
    if stats['active_entities']:
        print(f"  - Entities tracked: {', '.join(stats['active_entities'][:3])}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print(f"""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   ✓ API INTEGRATION WORKING!                                  ║
║                                                                ║
║   Your conversation is using:                                 ║
║   • ConversationState (tracks active topic, entities)         ║
║   • ConversationalKnowledge (stores extracted facts)          ║
║   • Memory-first retrieval (no web search if sufficient)      ║
║   • Pronoun resolution (he → entity)                          ║
║   • Follow-up detection (is this a follow-up question?)       ║
║                                                                ║
║   Next steps:                                                 ║
║   1. Implement LLM response generation (see API_INTEGRATION_  ║
║      READY.md TODO 1)                                         ║
║   2. Integrate your research agent (see TODO 2)               ║
║   3. Test full conversation flow                              ║
║   4. Deploy to production                                     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

Conversation ID: {conv_id}

Check the database:
  SELECT * FROM conversation_state WHERE conversation_id = '{conv_id}';
  SELECT * FROM conversational_knowledge WHERE conversation_id = '{conv_id}';

Read more:
  • API_INTEGRATION_READY.md - Full integration guide
  • LAYERED_MEMORY_ARCHITECTURE.md - Design details
""")
