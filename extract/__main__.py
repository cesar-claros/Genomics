"""CLI entrypoint: `python -m extract [TEXT]` or pipe text via stdin.

Runs rsID + gene extraction on the input, looks up each variant, and prints
a JSON document with mentions and verdicts.

Examples:
  python -m extract "We found that APOE rs429358 is associated with AD."
  cat abstract.txt | python -m extract
"""

from __future__ import annotations

import json
import sys

from .combine import extract_and_lookup

text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
result = extract_and_lookup(text)
print(json.dumps(result.model_dump(), indent=2, default=str))
