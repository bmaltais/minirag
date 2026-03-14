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


import pickle

def needs_rebuild(
    index_path: str | Path,
    source: str | Path | list[str | Path],
    glob: str | None = None,
) -> bool:
    """Return True if the index is missing or any source file is newer than the index.

    ``source`` can be either:
    - a directory path (glob applied), or
    - an explicit list of file paths (glob is ignored).

    If ``glob`` is None and ``source`` is a directory, it attempts to read the
    glob used at build time from the index. Defaults to ``**/*.md`` if not found.

    Use this to implement auto-rebuild without loading the index first::

        # directory-based corpus
        if needs_rebuild(INDEX_PATH, REFS_DIR):
            r = Retriever(INDEX_PATH)
            r.build_from_directory(REFS_DIR)

        # scattered-file corpus
        files = collect_memory_files()
        if needs_rebuild(INDEX_PATH, files):
            r = Retriever(INDEX_PATH)
            r.build_from_files(files)
    """
    p = Path(index_path)
    if not p.exists():
        return True
    index_mtime = p.stat().st_mtime

    if isinstance(source, list):
        for f in source:
            f_path = Path(f)
            if not f_path.exists() or f_path.stat().st_mtime > index_mtime:
                return True
        return False

    if glob is None:
        try:
            with p.open("rb") as f:
                data = pickle.load(f)
                glob = data.get("glob")
        except Exception:
            pass
        if glob is None:
            glob = "**/*.md"

    return any(f.stat().st_mtime > index_mtime for f in Path(source).glob(glob))


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
    # Building / Appending
    # ------------------------------------------------------------------

    def add_text(self, text: str, source: str = "", metadata: dict | None = None) -> "Retriever":
        """Add raw text to the index. Creates a new index if none exists."""
        if self._index is None:
            self._index = MiniIndex()
        self._index.add_text(text, source=source, metadata=metadata)
        return self

    def add_file(self, path: str | Path, metadata: dict | None = None) -> "Retriever":
        """Add a file to the index. Creates a new index if none exists."""
        if self._index is None:
            self._index = MiniIndex()
        self._index.add_file(path, metadata=metadata)
        return self

    def add_directory(
        self, path: str | Path, glob: str = "**/*.md", metadata: dict | None = None
    ) -> "Retriever":
        """Add a directory to the index. Creates a new index if none exists."""
        if self._index is None:
            self._index = MiniIndex()
        self._index.add_directory(path, glob=glob, metadata=metadata)
        return self

    def build(self, embeddings: bool = False, save: bool = True) -> "Retriever":
        """(Re)build the index and optionally save to disk."""
        if self._index is None:
            raise RuntimeError("No index to build. Call add_text/file/directory first.")
        self._index.build()
        if embeddings:
            self._index.build_embeddings()
        if save and self._index_path:
            self._index.save(self._index_path)
        return self

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

    def query(
        self,
        text: str,
        top_k: int = 5,
        hybrid: bool = False,
        sources: list[str] | None = None,
    ) -> list[dict]:
        """
        Retrieve top_k relevant chunks for the given query.

        Args:
            hybrid:  Use BM25+embedding RRF fusion. Requires the index to have
                     been built with ``embeddings=True``.
            sources: Optional list of source identifiers to filter by.

        Raises RuntimeError if no index has been built or loaded.
        """
        if self._index is None:
            raise RuntimeError(
                "No index loaded. Call build_from_files() or build_from_directory() first, "
                "or pass a valid index_path to the constructor."
            )
        if hybrid:
            return self._index.hybrid_search(text, top_k=top_k, sources=sources)
        return self._index.search(text, top_k=top_k, sources=sources)

    def query_text(
        self,
        text: str,
        top_k: int = 5,
        hybrid: bool = False,
        sources: list[str] | None = None,
    ) -> str:
        """
        Convenience method — returns joined text of top results,
        suitable for injecting into an agent prompt as context.

        Args:
            hybrid:  Use BM25+embedding RRF fusion (requires embeddings built).
            sources: Optional list of source identifiers to filter by.
        """
        hits = self.query(text, top_k=top_k, hybrid=hybrid, sources=sources)
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
