"""CLI entrypoint: `python -m lookup <rsid>`."""

from __future__ import annotations

import json
import sys

from .combine import lookup_variant

query = sys.argv[1] if len(sys.argv) > 1 else "rs429358"
result = lookup_variant(query)
print(json.dumps(result.model_dump(), indent=2, default=str))
