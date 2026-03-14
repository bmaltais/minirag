# GitHub Copilot Instructions

> For full development context, see [CLAUDE.md](../CLAUDE.md) at the repo root.

## Quick Reference

**Always use `uv`** — never `pip` directly.

```bash
uv sync                        # install deps
uv run pytest tests/ -v        # run tests
uv run black minirag/ tests/   # format code
```

## Versioning — Required on Every PR

**Bump `pyproject.toml → [project] → version` in every PR that changes code.**

| Change type | Version part |
|-------------|-------------|
| Bug fix / docs / refactor | PATCH (0.1.**x**) |
| New feature, new public API | MINOR (0.**x**.0) |
| Breaking API change | MAJOR (**x**.0.0) |

The CI `version-check` job will fail if the version was not bumped vs. the main branch.

## Key Files

| File | Purpose |
|------|---------|
| `minirag/retriever.py` | Public high-level API (`Retriever`, `needs_rebuild`) |
| `minirag/index.py` | BM25Plus index — build / save / load / search |
| `minirag/hybrid.py` | EmbedIndex + RRF merge |
| `minirag/chunker.py` | Paragraph-based chunking with overlap |
| `minirag/cli.py` | `minirag index / query / stats` CLI |
| `pyproject.toml` | Version, deps, build config |
| `tests/test_smoke.py` | 25 smoke tests — run before committing |

## Public API Contract

Do not change signatures of `Retriever`, `needs_rebuild`, or CLI commands without a MAJOR version bump.
Hybrid features require `minirag[hybrid]` optional dependency — do not add `sentence-transformers` to core deps.
