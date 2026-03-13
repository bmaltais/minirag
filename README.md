# minirag

Minimal BM25 + hybrid retrieval for offline memory/context lookup. No LLM, no API key required.

## What it does

Indexes markdown files (or any text) into a BM25Plus index and optionally an embedding index, then retrieves the most relevant chunks for a query. Designed to feed context into agent prompts.

## Install

```bash
# BM25 only (pure Python, no download)
uv pip install git+https://github.com/berna/minirag.git

# With hybrid BM25 + embedding support (~80 MB model download on first use)
uv pip install "git+https://github.com/berna/minirag.git#egg=minirag[hybrid]"
```

## CLI

```bash
# Index a directory of markdown files
minirag index ./notes --glob "**/*.md"

# With embeddings for hybrid search
minirag index ./notes --embeddings

# Search
minirag query "project deadline" --top-k 5
minirag query "project deadline" --hybrid    # BM25 + embedding RRF

# Stats
minirag stats
```

## Python API

```python
from minirag import Retriever

# BM25 only
r = Retriever(".minirag/index.pkl")
r.build_from_directory("./notes")
hits = r.query("project deadline", top_k=5)
context = r.query_text("project deadline")   # formatted for agent prompts

# Hybrid BM25 + embedding (Reciprocal Rank Fusion)
r.build_from_directory("./notes", embeddings=True)
hits = r.query("project deadline", top_k=5, hybrid=True)
context = r.query_text("project deadline", hybrid=True)
```

## How it works

- **Chunking**: splits documents on paragraph boundaries with sliding-window overlap. `score_text` (core paragraph only) is used for BM25/embedding scoring to avoid IDF inflation from overlap context.
- **BM25Plus**: unlike BM25Okapi, BM25Plus IDF is always non-negative (`log((N+1)/df)`), making it reliable on small corpora.
- **Hybrid / RRF**: embeddings use `all-MiniLM-L6-v2` (CPU, ~80 MB). Scores are fused with Reciprocal Rank Fusion: `score = 1/(k+bm25_rank) + 1/(k+embed_rank)`. No normalisation needed.
- **Persistence**: BM25 index + optional embedding vectors saved together in a single pickle file.

## Architecture

```
minirag/
├── chunker.py    — paragraph chunking with overlap
├── index.py      — BM25Plus index (build/save/load/search/hybrid_search)
├── hybrid.py     — EmbedIndex (sentence-transformers) + rrf_merge()
├── retriever.py  — high-level Retriever API
└── cli.py        — minirag index / query / stats
```

## Tests

```bash
uv run pytest tests/ -v
# 20 tests: 12 BM25 + 8 hybrid (hybrid auto-skips without sentence-transformers)
```

## Optional dependency

```bash
uv pip install "minirag[hybrid]"   # sentence-transformers + numpy
```
