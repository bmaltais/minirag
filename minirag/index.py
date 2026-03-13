"""
BM25 index — build, persist, and search.

Optionally enhanced with embedding-based retrieval via build_embeddings() +
hybrid_search(). Pure BM25 paths have no dependency on sentence-transformers.
"""

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Plus

from .chunker import Chunk, chunk_file, chunk_text


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-word characters. No NLTK needed."""
    return _TOKEN_RE.findall(text.lower())


class MiniIndex:
    """
    Holds a BM25 index over a collection of Chunks.

    Typical usage:
        idx = MiniIndex()
        idx.add_file("notes.md")
        idx.add_file("journal.txt")
        idx.build()
        results = idx.search("project deadlines", top_k=5)
        idx.save("my_index.pkl")

        # Later:
        idx2 = MiniIndex.load("my_index.pkl")
        results = idx2.search("project deadlines")
    """

    def __init__(self, max_chars: int = 512, overlap: int = 1):
        self.max_chars = max_chars
        self.overlap = overlap
        self._chunks: list[Chunk] = []
        self._bm25: BM25Plus | None = None
        self._embed = None  # EmbedIndex — built on demand via build_embeddings()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_text(self, text: str, source: str = "") -> int:
        """Index raw text. Returns number of chunks added."""
        chunks = chunk_text(
            text, source=source, max_chars=self.max_chars, overlap=self.overlap
        )
        self._chunks.extend(chunks)
        self._bm25 = None  # invalidate
        self._embed = None  # invalidate
        return len(chunks)

    def add_file(self, path: str | Path) -> int:
        """Index a file. Returns number of chunks added."""
        chunks = chunk_file(path, max_chars=self.max_chars, overlap=self.overlap)
        self._chunks.extend(chunks)
        self._bm25 = None
        self._embed = None  # invalidate
        return len(chunks)

    def add_directory(self, path: str | Path, glob: str = "**/*.md") -> dict[str, int]:
        """
        Recursively index all matching files under a directory.

        Returns {filename: chunk_count} mapping.
        """
        p = Path(path)
        results: dict[str, int] = {}
        for f in sorted(p.glob(glob)):
            if f.is_file():
                count = self.add_file(f)
                results[str(f)] = count
        return results

    def build(self) -> "MiniIndex":
        """(Re)build the BM25 index from current chunks. Called automatically by search()."""
        if not self._chunks:
            raise ValueError(
                "No documents indexed yet. Call add_file() or add_text() first."
            )
        # Use score_text (core only) so overlap context doesn't inflate df
        # BM25Plus guarantees non-negative IDF (unlike BM25Okapi on small corpora)
        corpus = [_tokenize(c.score_text) for c in self._chunks]
        self._bm25 = BM25Plus(corpus)
        return self

    def build_embeddings(self, model_name: str | None = None) -> "MiniIndex":
        """
        Build an embedding index alongside BM25 for use with hybrid_search().

        Downloads all-MiniLM-L6-v2 (~80 MB) on first call; cached thereafter.
        Runs on CPU — no GPU required.

        Args:
            model_name: Override the default sentence-transformers model.
        """
        if not self._chunks:
            raise ValueError(
                "No documents indexed yet. Call add_file() or add_text() first."
            )
        from .hybrid import EmbedIndex  # noqa: PLC0415

        self._embed = EmbedIndex(model_name=model_name)
        self._embed.build(self._chunks)
        return self

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Returns up to top_k chunks ranked by BM25 relevance.

        Each result dict:
            {
                "score": float,
                "text":  str,
                "source": str,
                "start_line": int,
            }
        """
        if self._bm25 is None:
            self.build()

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Efficiently find top-k using numpy argpartition
        if len(scores) > top_k:
            # Partition so that the top_k largest elements are at the end
            idx = np.argpartition(scores, -top_k)[-top_k:]
            # Sort only the top_k elements
            ranked = idx[np.argsort(scores[idx])[::-1]]
        else:
            ranked = np.argsort(scores)[::-1]

        # BM25Plus gives every chunk a floor score of idf(t)*delta even with tf=0.
        # Compute the true no-match baseline so we only return chunks with actual hits.
        no_match = sum(self._bm25.idf.get(t, 0) * self._bm25.delta for t in tokens)
        return [
            {
                "score": float(scores[i]),
                "text": self._chunks[i].text,
                "source": self._chunks[i].source,
                "start_line": self._chunks[i].start_line,
            }
            for i in ranked
            if scores[i] > no_match
        ]

    def hybrid_search(self, query: str, top_k: int = 5, k: int = 60) -> list[dict]:
        """
        Hybrid BM25 + embedding retrieval fused with Reciprocal Rank Fusion.

        Requires build_embeddings() to have been called first.

        Each ranker contributes a ranked list of candidates (3× top_k each).
        RRF combines them without score normalisation:
            final_score = 1/(k + bm25_rank) + 1/(k + embed_rank)

        Args:
            query:  Search query string.
            top_k:  Number of results to return.
            k:      RRF smoothing constant (default 60).

        Returns:
            List of hit dicts with an added ``rrf_score`` field, sorted by
            descending RRF score.
        """
        if self._embed is None:
            raise RuntimeError(
                "Embedding index not built. Call build_embeddings() first."
            )
        from .hybrid import rrf_merge  # noqa: PLC0415

        candidate_k = max(top_k * 3, 20)
        bm25_hits = self.search(query, top_k=candidate_k)
        embed_hits = self._embed.search(query, top_k=candidate_k)
        return rrf_merge(bm25_hits, embed_hits, k=k, top_k=top_k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the index to disk (pickle). Includes embeddings if built."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "chunks": self._chunks,
            "max_chars": self.max_chars,
            "overlap": self.overlap,
        }
        if self._embed is not None:
            data["embed_state"] = self._embed.get_state()
        with p.open("wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "MiniIndex":
        """Load a previously saved index, restoring embeddings if present."""
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        idx = cls(max_chars=data["max_chars"], overlap=data["overlap"])
        idx._chunks = data["chunks"]
        if "embed_state" in data:
            from .hybrid import EmbedIndex  # noqa: PLC0415

            idx._embed = EmbedIndex.from_state(data["embed_state"], idx._chunks)
        return idx

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "chunks": len(self._chunks),
            "sources": len({c.source for c in self._chunks}),
            "built": self._bm25 is not None,
            "embeddings": self._embed is not None,
        }
