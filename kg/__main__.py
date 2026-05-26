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

# Load code/.env BEFORE other imports so env vars consumed at module-load
# time (e.g. NCBI_EMAIL in lookup/clinvar.py) see the .env-supplied values.
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from dataclasses import asdict  # noqa: E402

from extract.combine import extract_and_lookup  # noqa: E402
from lookup.combine import lookup_variant  # noqa: E402

from .client import Neo4jClient  # noqa: E402
from .load import load_verdicts  # noqa: E402

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
