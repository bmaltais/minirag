"""
Embedding index and Reciprocal Rank Fusion for hybrid BM25+embedding retrieval.

Import cost: sentence-transformers (~80 MB, CPU-only) is lazy-loaded only when
EmbedIndex.build() is first called. Pure BM25 paths incur no import overhead.

RRF formula:  final_score = 1/(k + bm25_rank) + 1/(k + embed_rank)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .chunker import Chunk


def _cosine_sim(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """Batch cosine similarity: query (D,) vs corpus (N, D) → (N,)."""
    q_norm = np.linalg.norm(query_vec)
    if q_norm < 1e-9:
        return np.zeros(len(corpus_vecs))
    q = query_vec / q_norm
    norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (corpus_vecs / norms) @ q


def rrf_merge(
    bm25_hits: list[dict],
    embed_hits: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over two ranked hit lists.

    Score for each chunk: sum of 1/(k + rank + 1) across the lists it appears in.
    Chunks are keyed by (source, start_line). A chunk present in both lists
    accumulates contributions from each, naturally up-ranking agreed results.

    Args:
        bm25_hits:   Ranked hits from BM25 search (highest score first).
        embed_hits:  Ranked hits from embedding search (highest score first).
        k:           RRF smoothing constant (default 60, standard in the literature).
        top_k:       Number of results to return.

    Returns:
        List of hit dicts sorted by descending RRF score, each with an added
        ``rrf_score`` field.
    """
    all_hits: dict[tuple, dict] = {}
    rrf_scores: dict[tuple, float] = {}

    for rank, hit in enumerate(bm25_hits):
        key = (hit["source"], hit["start_line"])
        all_hits[key] = hit
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    for rank, hit in enumerate(embed_hits):
        key = (hit["source"], hit["start_line"])
        all_hits[key] = hit
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    top_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    return [{**all_hits[key], "rrf_score": rrf_scores[key]} for key in top_keys]


class EmbedIndex:
    """
    Cosine-similarity index using sentence-transformers all-MiniLM-L6-v2.

    The model (~80 MB) is downloaded on first use and cached by the
    sentence-transformers library. Inference runs on CPU — no GPU required.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._vecs: np.ndarray | None = None  # shape (N, D)
        self._chunks: list[Chunk] = []
        self._model = None  # lazy-loaded

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for hybrid search. "
                    "Install it with: uv pip install 'minirag[hybrid]'"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, chunks: list[Chunk]) -> "EmbedIndex":
        """Encode all chunks using score_text (same field BM25 uses)."""
        model = self._get_model()
        self._chunks = chunks
        texts = [c.score_text for c in chunks]
        self._vecs = model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return self

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return top_k chunks ranked by cosine similarity to the query."""
        if self._vecs is None:
            raise RuntimeError("EmbedIndex not built. Call build() first.")
        model = self._get_model()
        q_vec = model.encode([query], convert_to_numpy=True)[0]
        sims = _cosine_sim(q_vec, self._vecs)
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {
                "score": float(sims[i]),
                "text": self._chunks[i].text,
                "source": self._chunks[i].source,
                "start_line": self._chunks[i].start_line,
            }
            for i in ranked_idx
        ]

    # Serialization helpers — called by MiniIndex.save / MiniIndex.load

    def get_state(self) -> dict:
        return {"vecs": self._vecs, "model_name": self.model_name}

    @classmethod
    def from_state(cls, state: dict, chunks: list[Chunk]) -> "EmbedIndex":
        obj = cls(model_name=state["model_name"])
        obj._vecs = state["vecs"]
        obj._chunks = chunks
        return obj
