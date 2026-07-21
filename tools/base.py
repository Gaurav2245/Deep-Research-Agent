"""
tools/base.py
Abstract interface that every search tool must satisfy.
This decouples the agent graph from any concrete search implementation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class SearchResult:
    """Normalised result returned by any search provider."""
    title: str
    url: str
    content: str
    score: float = 0.0
    raw_content: str | None = None
    published_date: str | None = None


@dataclass
class SearchResponse:
    query: str
    results: List[SearchResult] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    answer: str | None = None          # Tavily can return a direct answer


class BaseSearchTool(ABC):
    """
    Interface every search provider must implement.
    Swap providers by swapping this class — no graph code changes needed.
    """

    @abstractmethod
    def search(self, query: str, **kwargs) -> SearchResponse:
        """Run a web search and return a normalised SearchResponse."""

    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier."""
