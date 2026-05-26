"""CLI entrypoint: `python -m extract [TEXT]` or pipe text via stdin.

Runs rsID + gene extraction on the input, looks up each variant, and prints
a JSON document with mentions and verdicts.

Examples:
  python -m extract "We found that APOE rs429358 is associated with AD."
  cat abstract.txt | python -m extract
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
import sys  # noqa: E402

from .combine import extract_and_lookup  # noqa: E402

text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
result = extract_and_lookup(text)
print(json.dumps(result.model_dump(), indent=2, default=str))
