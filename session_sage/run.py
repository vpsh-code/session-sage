"""CLI entry point for session-sage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extract import DEFAULT_DB, load_all
from .classify import classify_all
from .graph import build_graph, to_json
from .viz import write_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine Copilot CLI sessions into a typed knowledge graph"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to session-store.db (default: ~/.copilot/session-store.db)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Projects" / "session-sage" / "output" / "session_sage_graph.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only include sessions on or after this ISO date (e.g. 2026-05-01)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write raw graph JSON to this path",
    )
    parser.add_argument(
        "--title",
        default="Session Sage — Knowledge Graph",
        help="HTML page title",
    )
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"❌  DB not found: {args.db}", file=sys.stderr)
        return 1

    print(f"📂  Loading sessions from {args.db} …")
    sessions = load_all(args.db, since=args.since)
    all_turns = [t for s in sessions for t in s.turns]
    print(f"    {len(sessions)} sessions · {len(all_turns)} turns")

    print("🔬  Classifying signals …")
    signals = classify_all(all_turns)
    corrections = sum(1 for s in signals if s.is_correction)
    preferences = sum(1 for s in signals if s.is_preference)
    print(f"    {corrections} correction turns · {preferences} preference turns")

    print("🕸️   Building knowledge graph …")
    nodes, edges = build_graph(sessions, signals)
    print(f"    {len(nodes)} nodes · {len(edges)} edges")

    graph_data = to_json(nodes, edges)

    if args.json_out:
        import json
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(graph_data, indent=2))
        print(f"📄  JSON → {args.json_out}")

    print(f"🎨  Rendering HTML → {args.output} …")
    write_html(graph_data, args.output, title=args.title)

    print(f"\n✅  Done!  Open in browser:")
    print(f"    open '{args.output}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
