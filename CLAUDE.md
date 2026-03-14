# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Project Overview

`minirag` is a minimal BM25 + hybrid (BM25+embedding) retrieval library. No LLM required.
Fully offline, no API keys. Used to retrieve context from large file collections for agent workflows.

**Package:** `minirag/` — chunker, index, hybrid, retriever, cli
**Tests:** `tests/test_smoke.py` (25 tests: 12 BM25 + 8 hybrid + 5 CLI)
**Version:** `pyproject.toml → [project] version` (semantic versioning)

## Development Workflow

Always use `uv` — never `pip` directly.

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Format code before committing
uv run black minirag/ tests/

# Build and verify package
uv build
```

## Versioning — REQUIRED on Every PR

**The version in `pyproject.toml` MUST be bumped in every PR that changes code.**

Version lives at: `pyproject.toml` → `[project]` → `version`

Follow [Semantic Versioning](https://semver.org/):
- `PATCH` (0.1.x) — bug fixes, doc updates, minor internal refactors
- `MINOR` (0.x.0) — new public API, new features (backwards-compatible)
- `MAJOR` (x.0.0) — breaking changes to public API

After merging, tag the release:
```bash
git tag v<version>
git push origin v<version>
```

**Do not merge a PR without a version bump.** The CI `version-check` job enforces this on non-documentation-only PRs.

## Architecture Notes

- **BM25Plus** (not BM25Okapi): avoids negative IDF on small corpora
- **score_text vs text**: chunks store `score_text` (core only, for scoring) and `text` (core + overlap, for display)
- **No-match baseline filter**: filters chunks where BM25Plus gives non-zero scores even on misses
- **RRF merge**: `1/(k+bm25_rank) + 1/(k+embed_rank)`, k=60 default
- **Lazy model loading**: sentence-transformers imported only on first `EmbedIndex.build()` call
- **Embeddings persisted in same pickle** as BM25 state

## Do Not

- Use `pip` directly (use `uv`)
- Merge without bumping `pyproject.toml` version
- Store secrets in code
- Break the public API (`Retriever`, `needs_rebuild`) without a MAJOR version bump
- Skip tests before committing
