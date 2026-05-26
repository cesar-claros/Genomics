"""
MONDO disease/condition normalization via EBI's OLS API.

Resolves a free-text disease string (typically a ClinVar condition or a
GWAS Catalog trait without a MONDO URI) to a canonical MONDO ID and
label. Uses EBI's OLS4 search endpoint, which performs fuzzy matching
across many biomedical ontologies; we restrict to `ontology=mondo` so
only disease entries come back.

Each lookup is one HTTP call. Results are cached per surface form within
a Python process. For batch loads, this means ~one call per unique
disease string; identical names amortize to one fetch.

Why OLS rather than a local OBO file: the OBO is large, version-locked,
and matching against it requires a substantial fuzzy-search index of
its own. OLS provides exactly that as a service, with EBI maintaining
the index. Online dependency is acceptable for the same reasons as
ClinVar / GWAS Catalog.
"""

from __future__ import annotations

from functools import lru_cache

import requests

from .models import NormalizedDisease

_OLS_BASE = "https://www.ebi.ac.uk/ols4/api"


@lru_cache(maxsize=1024)
def _ols_search_mondo(name: str, timeout: int = 20) -> NormalizedDisease:
    """Query OLS4 for the top MONDO match for `name`. Returns a stub on miss/error."""
    if not name:
        return NormalizedDisease(surface=name)

    try:
        resp = requests.get(
            f"{_OLS_BASE}/search",
            params={
                "q": name,
                "ontology": "mondo",
                "exact": "false",
                "rows": 1,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])
    except (requests.RequestException, ValueError):
        return NormalizedDisease(surface=name)

    if not docs:
        return NormalizedDisease(surface=name)

    top = docs[0]
    return NormalizedDisease(
        surface=name,
        label=top.get("label"),
        mondo_id=top.get("obo_id"),
        iri=top.get("iri"),
        score=top.get("score"),
    )


def normalize_disease(name: str) -> NormalizedDisease:
    """Resolve a disease/condition name to its canonical MONDO entry.

    Returns a `NormalizedDisease` with `mondo_id`, `label`, `iri`,
    `score` populated on a hit; only `surface` populated on a miss or
    network error.

    Network errors are silently absorbed by design: the project's
    fail-soft contract says normalization should degrade to "no
    canonical ID" rather than raise.
    """
    return _ols_search_mondo(name)
