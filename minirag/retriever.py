"""
High-level retrieval API for agent / programmatic use.

This is the main entry point for skills agents:

    from minirag import Retriever

    r = Retriever("path/to/index.pkl")
    answer = r.query("What was the deadline for project X?")
    for hit in answer:
        print(hit["score"], hit["text"])
"""

from pathlib import Path

from .index import MiniIndex


class Retriever:
    """
    Thin wrapper around MiniIndex that handles index lifecycle.

    If index_path exists, it is loaded automatically.
    Otherwise, call build_from_files() or build_from_directory() to create it.
    """

    def __init__(self, index_path: str | Path | None = None):
        self._index_path = Path(index_path) if index_path else None
        self._index: MiniIndex | None = None

        if self._index_path and self._index_path.exists():
            self._index = MiniIndex.load(self._index_path)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build_from_files(
        self,
        files: list[str | Path],
        max_chars: int = 512,
        overlap: int = 1,
        save: bool = True,
        embeddings: bool = False,
    ) -> "Retriever":
        """Index a list of files and optionally persist.

        Args:
            embeddings: Also build an embedding index for hybrid_search().
                        Downloads all-MiniLM-L6-v2 (~80 MB) on first use.
        """
        self._index = MiniIndex(max_chars=max_chars, overlap=overlap)
        for f in files:
            self._index.add_file(f)
        self._index.build()
        if embeddings:
            self._index.build_embeddings()
        if save and self._index_path:
            self._index.save(self._index_path)
        return self

    def build_from_directory(
        self,
        directory: str | Path,
        glob: str = "**/*.md",
        max_chars: int = 512,
        overlap: int = 1,
        save: bool = True,
        embeddings: bool = False,
    ) -> "Retriever":
        """Index all matching files under a directory and optionally persist.

        Args:
            embeddings: Also build an embedding index for hybrid_search().
                        Downloads all-MiniLM-L6-v2 (~80 MB) on first use.
        """
        self._index = MiniIndex(max_chars=max_chars, overlap=overlap)
        self._index.add_directory(directory, glob=glob)
        self._index.build()
        if embeddings:
            self._index.build_embeddings()
        if save and self._index_path:
            self._index.save(self._index_path)
        return self

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int = 5, hybrid: bool = False) -> list[dict]:
        """
        Retrieve top_k relevant chunks for the given query.

        Args:
            hybrid: Use BM25+embedding RRF fusion. Requires the index to have
                    been built with ``embeddings=True``.

        Raises RuntimeError if no index has been built or loaded.
        """
        if self._index is None:
            raise RuntimeError(
                "No index loaded. Call build_from_files() or build_from_directory() first, "
                "or pass a valid index_path to the constructor."
            )
        if hybrid:
            return self._index.hybrid_search(text, top_k=top_k)
        return self._index.search(text, top_k=top_k)

    def query_text(self, text: str, top_k: int = 5, hybrid: bool = False) -> str:
        """
        Convenience method — returns joined text of top results,
        suitable for injecting into an agent prompt as context.

        Args:
            hybrid: Use BM25+embedding RRF fusion (requires embeddings built).
        """
        hits = self.query(text, top_k=top_k, hybrid=hybrid)
        if not hits:
            return ""
        return "\n\n---\n\n".join(
            f"[{h['source']} L{h['start_line']}] {h['text']}" for h in hits
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        if self._index is None:
            return {"status": "no index"}
        return self._index.stats()
