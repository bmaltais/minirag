"""minirag — minimal BM25-based retrieval. No LLM required."""

from .index import MiniIndex
from .retriever import Retriever

__all__ = ["MiniIndex", "Retriever"]
