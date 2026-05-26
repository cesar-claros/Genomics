"""
HGNC gene-symbol normalization.

Resolves a surface gene mention (e.g., "Apo E", "apolipoprotein E",
"APOE") to its approved HGNC symbol plus accession IDs by looking it up
in HGNC's `hgnc_complete_set.txt` TSV. The TSV is downloaded once on
first use and cached locally; subsequent calls in the same Python
process reuse an in-memory index.

Lookups are case-insensitive. The index is keyed by approved symbol
first, then by alias_symbol entries, then by prev_symbol entries, with
earlier entries winning. `alias_type` records which match path fired.

Configuration:
  HGNC_DATA_PATH  Path to a local TSV file. If unset, downloads to
                  ~/.cache/genomics-kg/hgnc_complete_set.txt on first call.

Why local TSV rather than the HGNC REST API: the TSV is ~15 MB,
downloads once, supports many thousands of lookups per second with no
network round-trips. The REST API is slower per call and adds a
hard external dependency for what is essentially static data.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

import requests

from .models import NormalizedGene

_DEFAULT_CACHE = "~/.cache/genomics-kg/hgnc_complete_set.txt"
_HGNC_URL = (
    "https://storage.googleapis.com/public-download-files/"
    "hgnc/tsv/tsv/hgnc_complete_set.txt"
)


def _data_path() -> Path:
    """Resolve the configured TSV path, expanding ~ and env vars."""
    return Path(os.environ.get("HGNC_DATA_PATH", _DEFAULT_CACHE)).expanduser()


def _download_hgnc(path: Path) -> None:
    """Fetch the HGNC TSV into `path`. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(_HGNC_URL, timeout=120)
    resp.raise_for_status()
    path.write_text(resp.text)


@lru_cache(maxsize=1)
def _load_hgnc_index() -> dict[str, dict[str, str | None]]:
    """Load the HGNC TSV into an alias-to-canonical index. Cached per process."""
    path = _data_path()
    if not path.exists():
        _download_hgnc(path)

    index: dict[str, dict[str, str | None]] = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                continue
            entry = {
                "symbol": symbol,
                "hgnc_id": (row.get("hgnc_id") or "").strip() or None,
                "entrez_id": (row.get("entrez_id") or "").strip() or None,
                "ensembl_id": (row.get("ensembl_gene_id") or "").strip() or None,
            }

            # Approved symbol always wins for its own key.
            index[symbol.upper()] = {**entry, "alias_type": "symbol"}

            # Aliases and previous symbols are pipe-separated in the TSV.
            for raw, kind in (
                (row.get("alias_symbol") or "", "alias"),
                (row.get("prev_symbol") or "", "previous"),
            ):
                stripped = raw.strip().strip('"')
                if not stripped:
                    continue
                for token in stripped.split("|"):
                    key = token.strip().upper()
                    if not key or key in index:
                        # Don't shadow approved-symbol matches with weaker links.
                        continue
                    index[key] = {**entry, "alias_type": kind}

    return index


def normalize_gene(surface: str) -> NormalizedGene:
    """Resolve a gene-surface mention to canonical HGNC info.

    Match precedence (case-insensitive):
      1. Approved HGNC symbol.
      2. Alias symbol from `alias_symbol`.
      3. Previous symbol from `prev_symbol`.

    No match returns a NormalizedGene with only the surface populated and
    `alias_type=None`.

    Free-text full names like "apolipoprotein E" are not in HGNC's alias
    columns by default and will currently miss. That's acceptable for v1;
    a future iteration could add full-name matching via the `name` column.
    """
    if not surface:
        return NormalizedGene(surface=surface)
    index = _load_hgnc_index()
    entry = index.get(surface.strip().upper())
    if entry is None:
        return NormalizedGene(surface=surface)
    return NormalizedGene(
        surface=surface,
        symbol=entry["symbol"],
        hgnc_id=entry["hgnc_id"],
        entrez_id=entry["entrez_id"],
        ensembl_id=entry["ensembl_id"],
        alias_type=entry["alias_type"],
    )
