#!/usr/bin/env python3
"""
Quick smoke test for the Deep Research Agent API.

Exercises the core endpoints against a running server:
    python -m uvicorn api.main:app --reload
    python quick_start.py

Usage:
    python quick_start.py ["your research question"]
"""
import os
import sys

import requests

API = "http://localhost:8000/api/v1"

session = requests.Session()
if os.getenv("API_KEY"):
    session.headers.update({"X-API-Key": os.environ["API_KEY"]})


def check_health() -> None:
    print("[1] Checking API health...")
    r = session.get("http://localhost:8000/health")
    r.raise_for_status()
    print(f"    OK: {r.json()}")


def create_conversation() -> str:
    print("\n[2] Creating a conversation...")
    r = session.post(f"{API}/conversations", json={"title": "Quick Start Test"})
    r.raise_for_status()
    conv = r.json()
    print(f"    Created conversation: {conv['id']}")
    return conv["id"]


def query_conversation(conversation_id: str, query: str) -> None:
    print(f"\n[3] Sending query: {query!r}")
    r = session.post(
        f"{API}/conversations/{conversation_id}/query",
        json={"query": query, "conversation_id": conversation_id},
    )
    r.raise_for_status()
    result = r.json()
    print(f"    Response ({result['elapsed_ms']:.0f}ms):")
    print(f"    {result['content'][:500]}")


def main() -> None:
    query = " ".join(sys.argv[1:]) or "What are the latest RBI monetary policy decisions?"
    try:
        check_health()
        conv_id = create_conversation()
        query_conversation(conv_id, query)
        print("\nAll checks passed.")
    except requests.exceptions.ConnectionError:
        print("\nCould not reach the API. Start it first with:")
        print("    uvicorn api.main:app --reload")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\nRequest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
