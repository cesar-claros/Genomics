"""CLI entrypoint: `python -m normalize <surface_or_rsid>`.

Auto-routes by input shape:
  - rsID -> looks up the variant and normalizes all its diseases.
  - Other text -> tries gene normalization first; falls back to disease
    if no HGNC match.

Examples:
  python -m normalize APOE
  python -m normalize "Apo E"
  python -m normalize "Alzheimer disease"
  python -m normalize rs429358
"""

from __future__ import annotations

# Load code/.env so HGNC_DATA_PATH overrides and the lookup stage's NCBI
# env vars (used by the rsID path) are visible.
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

from lookup.combine import lookup_variant  # noqa: E402

from .combine import normalize_verdict_diseases  # noqa: E402
from .diseases import normalize_disease  # noqa: E402
from .genes import normalize_gene  # noqa: E402

_RSID_RE = re.compile(r"^\s*rs\d+\s*$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="normalize",
        description="Normalize a gene symbol/alias, a disease name, "
                    "or all diseases in a variant verdict.",
    )
    parser.add_argument(
        "text",
        help="A gene symbol/alias, a disease name, or an rsID.",
    )
    parser.add_argument("--disease", action="store_true", help="Force disease lookup.")
    parser.add_argument("--gene", action="store_true", help="Force gene lookup.")
    args = parser.parse_args()

    if _RSID_RE.match(args.text):
        rsid = args.text.strip().lower()
        verdict = lookup_variant(rsid)
        normalized = normalize_verdict_diseases(verdict)
        payload = {
            "rsid": rsid,
            "n_diseases": len(normalized),
            "diseases": [d.model_dump() for d in normalized],
        }
    elif args.disease:
        payload = normalize_disease(args.text).model_dump()
    elif args.gene:
        payload = normalize_gene(args.text).model_dump()
    else:
        # Auto: try gene first, fall back to disease.
        g = normalize_gene(args.text)
        if g.symbol is not None:
            payload = {"resolved_as": "gene", **g.model_dump()}
        else:
            d = normalize_disease(args.text)
            payload = {"resolved_as": "disease", **d.model_dump()}

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
