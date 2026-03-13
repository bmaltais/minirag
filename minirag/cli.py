"""
minirag CLI

Commands:
    minirag index  <path> [options]   — build/update an index
    minirag query  <query> [options]  — search an existing index
    minirag stats  [options]          — show index info
"""

import argparse
import json
import sys
from pathlib import Path

from .index import MiniIndex
from .retriever import Retriever

DEFAULT_INDEX = ".minirag/index.pkl"


def cmd_index(args: argparse.Namespace) -> None:
    index_path = Path(args.index)

    # Load existing index to add to it, or start fresh
    if index_path.exists() and not args.reset:
        print(f"Loading existing index from {index_path} ...")
        idx = MiniIndex.load(index_path)
    else:
        idx = MiniIndex(max_chars=args.max_chars, overlap=args.overlap)

    target = Path(args.path)
    if target.is_file():
        count = idx.add_file(target)
        print(f"  Indexed {target}: {count} chunks")
    elif target.is_dir():
        results = idx.add_directory(target, glob=args.glob)
        total = sum(results.values())
        for f, n in results.items():
            print(f"  {f}: {n} chunks")
        print(f"Total: {total} chunks from {len(results)} files")
    else:
        print(f"Error: {target} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    idx.build()
    if args.embeddings:
        print("Building embedding index (downloads ~80 MB model on first run) ...")
        idx.build_embeddings()
    idx.save(index_path)
    s = idx.stats()
    suffix = "  [+embeddings]" if s["embeddings"] else ""
    print(f"Index saved to {index_path}  ({s['chunks']} total chunks){suffix}")


def cmd_query(args: argparse.Namespace) -> None:
    index_path = Path(args.index)
    if not index_path.exists():
        print(
            f"Error: no index at {index_path}. Run `minirag index` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    retriever = Retriever(index_path)
    hits = retriever.query(args.query, top_k=args.top_k, hybrid=args.hybrid)

    if not hits:
        print("No results found.")
        return

    if args.json:
        print(json.dumps(hits, indent=2))
        return

    score_key = "rrf_score" if args.hybrid else "score"
    for i, hit in enumerate(hits, 1):
        print(f"\n{'-'*60}")
        score_val = hit.get(score_key, hit.get("score", 0))
        print(
            f"#{i}  {score_key}={score_val:.4f}  {hit['source']}  L{hit['start_line']}"
        )
        print(f"{'-'*60}")
        text = hit["text"]
        if args.truncate and len(text) > args.truncate:
            text = text[: args.truncate] + " ..."
        print(text)


def cmd_stats(args: argparse.Namespace) -> None:
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"No index at {index_path}")
        return
    idx = MiniIndex.load(index_path)
    s = idx.stats()
    print(f"Index: {index_path}")
    print(f"  Chunks:     {s['chunks']}")
    print(f"  Sources:    {s['sources']}")
    print(f"  Embeddings: {'yes' if s['embeddings'] else 'no'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="minirag",
        description="Minimal BM25 retrieval — no LLM required",
    )
    parser.add_argument(
        "--index",
        "-i",
        default=DEFAULT_INDEX,
        help=f"Path to index file (default: {DEFAULT_INDEX})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- index ---
    p_idx = sub.add_parser("index", help="Index files or a directory")
    p_idx.add_argument("path", help="File or directory to index")
    p_idx.add_argument(
        "--glob",
        default="**/*.md",
        help="Glob pattern for directories (default: **/*.md)",
    )
    p_idx.add_argument(
        "--max-chars", type=int, default=512, help="Max chars per chunk (default: 512)"
    )
    p_idx.add_argument(
        "--overlap", type=int, default=1, help="Adjacent paragraph overlap (default: 1)"
    )
    p_idx.add_argument(
        "--reset", action="store_true", help="Discard existing index before indexing"
    )
    p_idx.add_argument(
        "--embeddings",
        action="store_true",
        help="Also build embedding index for hybrid search (downloads ~80 MB on first use)",
    )

    # --- query ---
    p_qry = sub.add_parser("query", help="Search the index")
    p_qry.add_argument("query", help="Search query")
    p_qry.add_argument(
        "--top-k", "-k", type=int, default=5, help="Number of results (default: 5)"
    )
    p_qry.add_argument(
        "--truncate",
        "-t",
        type=int,
        default=300,
        help="Truncate result text to N chars (0=off)",
    )
    p_qry.add_argument("--json", action="store_true", help="Output as JSON")
    p_qry.add_argument(
        "--hybrid",
        action="store_true",
        help="Use BM25+embedding RRF fusion (requires index built with --embeddings)",
    )

    # --- stats ---
    sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "stats":
        cmd_stats(args)
