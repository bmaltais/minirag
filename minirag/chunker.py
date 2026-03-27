"""
Splits documents into overlapping text chunks for indexing.

Strategy:
  1. Split on paragraph boundaries (blank lines) first.
  2. If a paragraph exceeds max_chars, split further on sentence boundaries.
  3. Slide a window over adjacent paragraphs to produce overlapping chunks.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    text: str  # display text: core + overlap context (shown to caller)
    score_text: str  # BM25 input: core paragraph only (prevents IDF pollution)
    source: str  # file path or identifier
    start_line: int  # approximate line number in source
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter (no NLTK dependency)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _split_paragraphs(text: str) -> list[tuple[int, str]]:
    """Return (start_line, paragraph_text) pairs."""
    lines = text.splitlines()
    paragraphs = []
    buf: list[str] = []
    start = 0

    for i, line in enumerate(lines):
        if line.strip() == "":
            if buf:
                paragraphs.append((start, " ".join(buf)))
                buf = []
        else:
            if not buf:
                start = i
            buf.append(line.strip())

    if buf:
        paragraphs.append((start, " ".join(buf)))

    return paragraphs


def chunk_text(
    text: str,
    source: str = "",
    max_chars: int = 512,
    overlap: int = 1,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Args:
        text:      raw document text
        source:    label for the chunk (filename, etc.)
        max_chars: max characters per chunk before sub-splitting
        overlap:   number of adjacent paragraphs to include as overlap context
        metadata:  optional dictionary of metadata to attach to each chunk
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    # Sub-split oversized paragraphs
    units: list[tuple[int, str]] = []
    for line_no, para in paragraphs:
        if len(para) <= max_chars:
            units.append((line_no, para))
        else:
            sentences = _split_sentences(para)
            buf = ""
            for sent in sentences:
                if len(buf) + len(sent) + 1 > max_chars and buf:
                    units.append((line_no, buf.strip()))
                    buf = sent
                else:
                    buf = (buf + " " + sent).strip() if buf else sent
            if buf:
                units.append((line_no, buf.strip()))

    # Sliding window with overlap
    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        start_line, core = units[i]
        parts = [core]

        # Look back for overlap context (display only)
        for j in range(max(0, i - overlap), i):
            parts.insert(0, units[j][1])

        # Look forward for overlap context (display only)
        for j in range(i + 1, min(len(units), i + 1 + overlap)):
            parts.append(units[j][1])

        display = " ".join(parts)
        # score_text uses only the core paragraph to avoid IDF pollution from overlap
        chunks.append(
            Chunk(
                text=display,
                score_text=core,
                source=source,
                start_line=start_line,
                metadata=dict(metadata or {}),
            )
        )
        i += 1

    return chunks


def chunk_file(
    path: str | Path, metadata: dict | None = None, **kwargs
) -> list[Chunk]:
    """Read a file and return its chunks.

    Automatically stores the file's modification time in chunk metadata under
    the ``mtime`` key (float, seconds since epoch). Caller-supplied metadata
    takes precedence and can override this value.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    # Seed metadata with mtime so incremental indexing can detect changes.
    meta: dict = {"mtime": p.stat().st_mtime}
    if metadata:
        meta.update(metadata)
    return chunk_text(text, source=str(p), metadata=meta, **kwargs)
