"""CLI entrypoint: `python -m kg <text-or-rsid>`.

Looks up the variant(s) and upserts them into Neo4j. Reports node/edge
counts as JSON.

Examples:
  python -m kg rs429358
  python -m kg "We genotyped APOE rs429358 in patients with AD."
  cat passage.txt | python -m kg

Environment:
  NEO4J_URI       (required, e.g. bolt://localhost:7687)
  NEO4J_USER      (default: neo4j)
  NEO4J_PASSWORD  (required)
  NEO4J_DATABASE  (optional, default DB)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict

from extract.combine import extract_and_lookup
from lookup.combine import lookup_variant

from .client import Neo4jClient
from .load import load_verdicts

_RSID_ONLY_RE = re.compile(r"^\s*rs\d+\s*$", re.IGNORECASE)


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    if not text:
        print("usage: python -m kg <text-or-rsid>", file=sys.stderr)
        sys.exit(2)

    if _RSID_ONLY_RE.match(text):
        verdicts = [lookup_variant(text.strip().lower())]
    else:
        verdicts = extract_and_lookup(text).verdicts

    try:
        client = Neo4jClient.from_env()
    except ValueError as e:
        print(f"Neo4j config error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with client:
            stats = load_verdicts(client, verdicts)
    except Exception as e:  # noqa: BLE001 - surface the actual failure
        print(f"Neo4j load failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    payload = {
        "input": text,
        "verdicts_loaded": len(verdicts),
        "rsids": [v.rsid or v.query for v in verdicts],
        "merged": asdict(stats),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
