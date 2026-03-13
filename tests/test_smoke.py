"""Smoke tests for minirag — no LLM required."""

import tempfile
from pathlib import Path

import pytest

from minirag import MiniIndex, Retriever
from minirag.chunker import chunk_text, chunk_file

# ------------------------------------------------------------------
# Chunker
# ------------------------------------------------------------------

SAMPLE = """\
The quick brown fox jumps over the lazy dog.
This is a simple sentence.

A second paragraph with completely different content.
It discusses cats and mice and cheese.

Third paragraph about deadlines and project management.
The project deadline is next Friday.
"""


def test_chunk_text_basic():
    chunks = chunk_text(SAMPLE, source="test")
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text
        assert c.source == "test"


def test_chunk_text_overlap():
    chunks = chunk_text(SAMPLE, overlap=1)
    # With overlap, each chunk should mention content from adjacent paragraphs
    core_chunks = chunk_text(SAMPLE, overlap=0)
    for ov, core in zip(chunks, core_chunks):
        assert len(ov.text) >= len(core.text)


def test_chunk_file(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text(SAMPLE)
    chunks = chunk_file(f)
    assert len(chunks) >= 2
    assert chunks[0].source == str(f)


# ------------------------------------------------------------------
# MiniIndex
# ------------------------------------------------------------------


def test_index_build_and_search():
    idx = MiniIndex()
    idx.add_text(SAMPLE, source="sample")
    idx.build()

    results = idx.search("project deadline", top_k=3)
    assert results, "Expected at least one result"
    # Top result should contain deadline-related content
    assert any("deadline" in r["text"].lower() for r in results)


def test_index_no_results_for_gibberish():
    idx = MiniIndex()
    idx.add_text(SAMPLE, source="sample")
    results = idx.search("xyzzy frobnicator zork", top_k=5)
    assert results == []  # all scores are 0


def test_index_save_load(tmp_path):
    idx = MiniIndex()
    idx.add_text(SAMPLE, source="sample")
    idx.build()

    path = tmp_path / "index.pkl"
    idx.save(path)

    idx2 = MiniIndex.load(path)
    results = idx2.search("cats mice cheese", top_k=3)
    assert results
    assert any("cats" in r["text"].lower() for r in results)


def test_index_add_directory(tmp_path):
    (tmp_path / "a.md").write_text("Alpha document with apples and oranges.")
    (tmp_path / "b.md").write_text("Beta document about bananas and grapes.")
    (tmp_path / "c.txt").write_text("This should not be indexed by default glob.")

    idx = MiniIndex()
    counts = idx.add_directory(tmp_path, glob="*.md")
    assert len(counts) == 2
    idx.build()

    results = idx.search("apples")
    assert results
    assert "a.md" in results[0]["source"]


def test_index_stats():
    idx = MiniIndex()
    idx.add_text(SAMPLE, source="s1")
    idx.add_text("Another document entirely.", source="s2")
    s = idx.stats()
    assert s["sources"] == 2
    assert s["chunks"] >= 2


# ------------------------------------------------------------------
# Retriever
# ------------------------------------------------------------------


def test_retriever_build_from_files(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text(SAMPLE)
    idx_path = tmp_path / "index.pkl"

    r = Retriever(idx_path)
    r.build_from_files([f])

    hits = r.query("project deadline", top_k=3)
    assert hits
    assert any("deadline" in h["text"].lower() for h in hits)


def test_retriever_query_text(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text(SAMPLE)
    idx_path = tmp_path / "index.pkl"

    r = Retriever(idx_path)
    r.build_from_files([f])

    ctx = r.query_text("cats mice")
    assert "cats" in ctx.lower()


def test_retriever_load_from_disk(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text(SAMPLE)
    idx_path = tmp_path / "index.pkl"

    Retriever(idx_path).build_from_files([f])

    # Reload from disk
    r2 = Retriever(idx_path)
    hits = r2.query("fox jumps", top_k=3)
    assert hits


def test_retriever_raises_without_index():
    r = Retriever()  # no index path, no build
    with pytest.raises(RuntimeError):
        r.query("anything")


# ------------------------------------------------------------------
# Hybrid: Reciprocal Rank Fusion (no model required)
# ------------------------------------------------------------------


def test_rrf_merge_basic():
    pytest.importorskip("numpy")
    from minirag.hybrid import rrf_merge

    bm25 = [
        {"source": "a.md", "start_line": 0, "score": 0.9, "text": "alpha content"},
        {"source": "b.md", "start_line": 0, "score": 0.5, "text": "beta content"},
    ]
    embed = [
        {"source": "b.md", "start_line": 0, "score": 0.8, "text": "beta content"},
        {"source": "a.md", "start_line": 0, "score": 0.3, "text": "alpha content"},
    ]
    results = rrf_merge(bm25, embed, k=60, top_k=2)
    assert len(results) == 2
    assert all("rrf_score" in r for r in results)
    assert all(r["rrf_score"] > 0 for r in results)


def test_rrf_merge_single_list():
    pytest.importorskip("numpy")
    from minirag.hybrid import rrf_merge

    bm25 = [{"source": "x.md", "start_line": 1, "score": 0.5, "text": "only bm25"}]
    results = rrf_merge(bm25, [], top_k=3)
    assert len(results) == 1
    assert results[0]["rrf_score"] > 0


# ------------------------------------------------------------------
# Hybrid: Embedding index + full pipeline
# ------------------------------------------------------------------


def test_embed_index_search():
    pytest.importorskip("sentence_transformers")
    from minirag.hybrid import EmbedIndex

    from minirag.chunker import chunk_text as _chunk

    chunks = _chunk(SAMPLE, source="test")
    ei = EmbedIndex()
    ei.build(chunks)
    results = ei.search("project deadline", top_k=3)
    assert results
    assert results[0]["score"] > 0


def test_hybrid_search_basic():
    pytest.importorskip("sentence_transformers")
    idx = MiniIndex()
    idx.add_text(SAMPLE, source="sample")
    idx.build()
    idx.build_embeddings()
    results = idx.hybrid_search("project deadline", top_k=3)
    assert results
    assert all("rrf_score" in r for r in results)
    assert any("deadline" in r["text"].lower() for r in results)


def test_hybrid_search_no_embed_raises():
    idx = MiniIndex()
    idx.add_text(SAMPLE, source="sample")
    idx.build()
    with pytest.raises(RuntimeError, match="build_embeddings"):
        idx.hybrid_search("anything")


def test_hybrid_search_synonym():
    """Embedding surfaces semantic synonyms that BM25 misses by exact term."""
    pytest.importorskip("sentence_transformers")
    doc = """\
Model selection determines which AI system to use for a task.
The configuration field specifies model_id and temperature.

Unrelated content about databases, SQL, and query optimisation.
"""
    idx = MiniIndex()
    idx.add_text(doc, source="config.md")
    idx.build()
    idx.build_embeddings()
    results = idx.hybrid_search("choosing a model", top_k=2)
    assert results
    assert any("model" in r["text"].lower() for r in results)


def test_index_save_load_with_embeddings(tmp_path):
    """Embeddings survive a pickle save/load round-trip."""
    pytest.importorskip("sentence_transformers")
    idx = MiniIndex()
    idx.add_text(SAMPLE, source="sample")
    idx.build()
    idx.build_embeddings()

    path = tmp_path / "index.pkl"
    idx.save(path)

    idx2 = MiniIndex.load(path)
    assert idx2.stats()["embeddings"]
    results = idx2.hybrid_search("cats mice", top_k=3)
    assert results


def test_retriever_hybrid(tmp_path):
    pytest.importorskip("sentence_transformers")
    f = tmp_path / "notes.md"
    f.write_text(SAMPLE)
    idx_path = tmp_path / "index.pkl"

    r = Retriever(idx_path)
    r.build_from_files([f], embeddings=True)
    hits = r.query("project deadline", top_k=3, hybrid=True)
    assert hits
    assert any("deadline" in h["text"].lower() for h in hits)
