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
        self.glob: str | None = None
        self._chunks: list[Chunk] = []
        self._bm25: BM25Plus | None = None
        self._embed = None  # EmbedIndex — built on demand via build_embeddings()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_text(self, text: str, source: str = "", metadata: dict | None = None) -> int:
        """Index raw text. Returns number of chunks added."""
        chunks = chunk_text(
            text,
            source=source,
            max_chars=self.max_chars,
            overlap=self.overlap,
            metadata=metadata,
        )
        self._chunks.extend(chunks)
        self._bm25 = None  # invalidate
        self._embed = None  # invalidate
        return len(chunks)

    def add_file(self, path: str | Path, metadata: dict | None = None) -> int:
        """Index a file. Returns number of chunks added."""
        chunks = chunk_file(
            path, max_chars=self.max_chars, overlap=self.overlap, metadata=metadata
        )
        self._chunks.extend(chunks)
        self._bm25 = None
        self._embed = None  # invalidate
        return len(chunks)

    def add_directory(
        self,
        path: str | Path,
        glob: str = "**/*.md",
        metadata: dict | None = None,
    ) -> dict[str, int]:
        """
        Recursively index all matching files under a directory.

        Returns {filename: chunk_count} mapping.
        """
        self.glob = glob
        p = Path(path)
        results: dict[str, int] = {}
        for f in sorted(p.glob(glob)):
            if f.is_file():
                count = self.add_file(f, metadata=metadata)
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

    def build_embeddings(
        self, model_name: str | None = None, device: str | None = None
    ) -> "MiniIndex":
        """
        Build an embedding index alongside BM25 for use with hybrid_search().

        Downloads all-MiniLM-L6-v2 (~80 MB) on first call; cached thereafter.
        Runs on CPU by default, but supports CUDA/MPS if a device is specified.

        Args:
            model_name: Override the default sentence-transformers model.
            device:     Device to run embeddings on (e.g. "cuda", "cpu", "mps").
        """
        if not self._chunks:
            raise ValueError(
                "No documents indexed yet. Call add_file() or add_text() first."
            )
        from .hybrid import EmbedIndex  # noqa: PLC0415

        self._embed = EmbedIndex(model_name=model_name, device=device)
        self._embed.build(self._chunks)
        return self


    def remove_file(self, path: str | Path) -> int:
        """Remove all chunks belonging to a file from the index.

        Also removes the corresponding rows from the embedding matrix if one
        has been built, keeping BM25 and embedding indexes in sync.

        Args:
            path: Path to the file whose chunks should be removed.

        Returns:
            Number of chunks removed (0 if the file was not indexed).

        Note:
            This invalidates the BM25 index. Call ``build()`` (and optionally
            ``build_embeddings()``) after making all mutations.
        """
        target = str(Path(path).resolve())
        keep = [
            i for i, c in enumerate(self._chunks)
            if str(Path(c.source).resolve()) != target
        ]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0

        self._chunks = [self._chunks[i] for i in keep]

        if self._embed is not None:
            self._embed.remove_rows(keep)

        self._bm25 = None  # invalidate - rebuild required
        return removed

    def update_file(
        self, path: str | Path, metadata: dict | None = None
    ) -> tuple[int, int]:
        """Replace a file's chunks with a fresh re-index.

        Equivalent to ``remove_file(path)`` followed by ``add_file(path)``,
        but expressed as a single atomic operation.

        Args:
            path:     Path to the file to re-index.
            metadata: Optional metadata dict to attach to the new chunks.

        Returns:
            ``(removed, added)`` - chunk counts for the old and new versions.

        Note:
            This invalidates the BM25 index and embedding matrix. Call
            ``build()`` and ``build_embeddings()`` after all mutations are done.
        """
        removed = self.remove_file(path)
        added = self.add_file(path, metadata=metadata)
        return removed, added

    def scan_directory(
        self,
        path: str | Path,
        glob: str = "**/*.md",
        mtime_tolerance: float = 1.0,
    ) -> dict[str, list]:
        """Compare files on disk with the current index state.

        Uses the ``mtime`` value stored in chunk metadata (populated
        automatically by ``chunk_file``) to detect which files have changed.

        Args:
            path:            Directory to scan.
            glob:            Glob pattern for matching files.
            mtime_tolerance: Files with ``disk_mtime - stored_mtime`` below
                             this threshold (seconds) are treated as unchanged.

        Returns:
            A dict with three keys:

            * ``"new"``     - ``list[Path]``: on disk, not yet indexed.
            * ``"changed"`` - ``list[Path]``: indexed, but newer on disk.
            * ``"deleted"`` - ``list[str]``:  indexed paths no longer on disk.

        Example::

            diff = idx.scan_directory("./docs")
            for f in diff["deleted"]:
                idx.remove_file(f)
            for f in diff["changed"]:
                idx.update_file(f)
            for f in diff["new"]:
                idx.add_file(f)
            idx.build()
        """
        p = Path(path)
        disk_files = {
            str(f.resolve()): f
            for f in p.glob(glob)
            if f.is_file()
        }
        indexed_paths = {str(Path(c.source).resolve()) for c in self._chunks}

        new = [f for rp, f in disk_files.items() if rp not in indexed_paths]
        deleted = [rp for rp in indexed_paths if rp not in disk_files]

        # Build per-source stored mtime (max across chunks for that source)
        stored_mtimes: dict[str, float] = {}
        for c in self._chunks:
            rp = str(Path(c.source).resolve())
            mt = c.metadata.get("mtime")
            if mt is not None:
                stored_mtimes[rp] = max(mt, stored_mtimes.get(rp, 0.0))

        changed = [
            f
            for rp, f in disk_files.items()
            if rp in indexed_paths
            and rp in stored_mtimes
            and f.stat().st_mtime > stored_mtimes[rp] + mtime_tolerance
        ]

        return {"new": new, "changed": changed, "deleted": deleted}

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 5, sources: list[str] | None = None
    ) -> list[dict]:
        """
        Returns up to top_k chunks ranked by BM25 relevance.

        Args:
            query:   Search query string.
            top_k:   Number of results to return.
            sources: Optional list of source identifiers to filter by.

        Each result dict:
            {
                "score": float,
                "text":  str,
                "source": str,
                "start_line": int,
                "metadata": dict,
            }
        """
        if self._bm25 is None:
            self.build()

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Apply source filtering if requested
        if sources is not None:
            # Efficiently mask scores for non-matching sources (substring match)
            for i, chunk in enumerate(self._chunks):
                if not any(s in chunk.source for s in sources):
                    scores[i] = -1e9  # Effectively ignore

        if len(scores) <= top_k:
            ranked = np.argsort(scores)[::-1]
        else:
            # Use argpartition for faster top-K selection
            indices = np.argpartition(scores, -top_k)[-top_k:]
            ranked = indices[np.argsort(scores[indices])[::-1]]

        # BM25Plus gives every chunk a floor score of idf(t)*delta even with tf=0.
        # Compute the true no-match baseline so we only return chunks with actual hits.
        no_match = sum(self._bm25.idf.get(t, 0) * self._bm25.delta for t in tokens)
        return [
            {
                "score": float(scores[i]),
                "text": self._chunks[i].text,
                "source": self._chunks[i].source,
                "start_line": self._chunks[i].start_line,
                "metadata": getattr(self._chunks[i], "metadata", {}),
            }
            for i in ranked
            if scores[i] > no_match
        ]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        k: int = 60,
        sources: list[str] | None = None,
        device: str | None = None,
    ) -> list[dict]:
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
        bm25_hits = self.search(query, top_k=candidate_k, sources=sources)
        embed_hits = self._embed.search(
            query, top_k=candidate_k, sources=sources, device=device
        )
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
            "glob": self.glob,
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
        idx.glob = data.get("glob")

        # Ensure backward compatibility for metadata
        for chunk in idx._chunks:
            if not hasattr(chunk, "metadata"):
                chunk.metadata = {}

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
            "fresh_build": self._bm25 is not None,
            "embeddings": self._embed is not None,
            "glob": self.glob,
        }
