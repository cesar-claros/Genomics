"""CLI entrypoint: `python -m lookup <rsid>`."""

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

from .combine import lookup_variant  # noqa: E402

query = sys.argv[1] if len(sys.argv) > 1 else "rs429358"
result = lookup_variant(query)
print(json.dumps(result.model_dump(), indent=2, default=str))
