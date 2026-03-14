
import pytest
from pathlib import Path
from minirag import MiniIndex, Retriever

SAMPLE = """
Paragraph 1. This is the first paragraph.
Paragraph 2. This is the second paragraph.
"""

def test_metadata_support():
    idx = MiniIndex()
    idx.add_text(SAMPLE, source="test", metadata={"category": "test_cat", "priority": 1})
    idx.build()

    results = idx.search("paragraph")
    assert len(results) > 0
    for r in results:
        assert "metadata" in r
        assert r["metadata"]["category"] == "test_cat"
        assert r["metadata"]["priority"] == 1

def test_source_filtering():
    idx = MiniIndex()
    idx.add_text("Content from source A", source="A")
    idx.add_text("Content from source B", source="B")
    idx.build()

    # Query without filter
    results = idx.search("content")
    sources = {r["source"] for r in results}
    assert "A" in sources
    assert "B" in sources

    # Query with filter
    results_a = idx.search("content", sources=["A"])
    for r in results_a:
        assert r["source"] == "A"

    results_b = idx.search("content", sources=["B"])
    for r in results_b:
        assert r["source"] == "B"

def test_retriever_enhanced_api(tmp_path):
    idx_path = tmp_path / "index.pkl"
    r = Retriever(idx_path)

    # Use new fluent-ish API
    r.add_text("First doc", source="doc1", metadata={"id": 1})
    r.add_text("Second doc", source="doc2", metadata={"id": 2})
    r.build(save=True)

    assert idx_path.exists()

    hits = r.query("doc", sources=["doc1"])
    assert len(hits) == 1
    assert hits[0]["source"] == "doc1"
    assert hits[0]["metadata"]["id"] == 1

def test_add_file_with_metadata(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("Some content")
    idx = MiniIndex()
    idx.add_file(f, metadata={"source_type": "file"})
    idx.build()
    results = idx.search("content")
    assert results[0]["metadata"]["source_type"] == "file"

def test_retriever_add_directory(tmp_path):
    (tmp_path / "a.md").write_text("Content A")
    idx_path = tmp_path / "index.pkl"
    r = Retriever(idx_path)
    r.add_directory(tmp_path)
    r.build()
    hits = r.query("Content")
    assert len(hits) > 0

def test_retriever_query_text_with_filter():
    r = Retriever()
    r.add_text("Hello from A", source="A")
    r.add_text("Hello from B", source="B")
    r.build()

    ctx_a = r.query_text("hello", sources=["A"])
    assert "from A" in ctx_a
    assert "from B" not in ctx_a

@pytest.mark.skipif(True, reason="Requires sentence-transformers")
def test_hybrid_source_filtering():
    # This is a placeholder to show intent; actual test would need model
    idx = MiniIndex()
    idx.add_text("Semantic match from A", source="A")
    idx.add_text("Semantic match from B", source="B")
    idx.build()
    idx.build_embeddings()

    results = idx.hybrid_search("semantic", sources=["A"])
    for r in results:
        assert r["source"] == "A"
