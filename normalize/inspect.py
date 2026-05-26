"""
Inspection tool for the normalization stage.

Given a surface form (gene symbol, alias, or disease name) or an rsID,
walks through the normalization process and prints every intermediate
result. No side effects beyond the network calls the resolvers
themselves perform.

Usage:
  python -m normalize.inspect APOE
  python -m normalize.inspect "Apo E"
  python -m normalize.inspect "Alzheimer disease"
  python -m normalize.inspect rs429358        # walks the verdict's diseases
"""

from __future__ import annotations

# Load code/.env before other imports so any downstream env-driven
# components (e.g., HGNC_DATA_PATH overrides, lookup NCBI vars when
# resolving a verdict) see the configured values.
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import argparse  # noqa: E402
import re  # noqa: E402

from lookup.combine import lookup_variant  # noqa: E402
from lookup.heuristics import is_disease_uri  # noqa: E402

from .combine import normalize_verdict_diseases  # noqa: E402
from .diseases import _ols_search_mondo, normalize_disease  # noqa: E402
from .genes import _data_path, _load_hgnc_index, normalize_gene  # noqa: E402

_RSID_RE = re.compile(r"^\s*rs\d+\s*$", re.IGNORECASE)


def _rule(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def _inspect_gene(surface: str) -> None:
    _rule(f"GENE NORMALIZATION (HGNC)")
    print(f"  Surface form: {surface!r}")
    print()

    path = _data_path()
    print(f"  HGNC TSV path: {path}")
    print(f"  TSV present:   {path.exists()}")
    print()
    if not path.exists():
        print("  First call will download HGNC's hgnc_complete_set.txt (~15 MB) into")
        print("  the path above. Override with HGNC_DATA_PATH if needed.")
        print()

    print("  Loading / reusing the in-memory alias index (cached per process)...")
    index = _load_hgnc_index()
    print(f"  Total alias keys in index: {len(index)}")
    print()

    key = surface.strip().upper()
    if key in index:
        entry = index[key]
        print(f"  Lookup key (uppercased): {key!r}")
        print(f"  Match found via alias_type={entry['alias_type']!r}:")
        for k, v in entry.items():
            print(f"    {k:12s} : {v!r}")
    else:
        print(f"  Lookup key (uppercased): {key!r}")
        print("  No match in approved symbol / alias / previous-symbol columns.")
        print("  Returns a NormalizedGene with only `surface` populated.")
    print()

    normalized = normalize_gene(surface)
    print("  Resulting NormalizedGene:")
    for k, v in normalized.model_dump().items():
        print(f"    {k:12s} : {v!r}")
    print()


def _inspect_disease(surface: str) -> None:
    _rule(f"DISEASE NORMALIZATION (MONDO via OLS4)")
    print(f"  Surface form: {surface!r}")
    print()

    print("  EBI OLS4 search:")
    print(f"    GET https://www.ebi.ac.uk/ols4/api/search")
    print(f"        ?q={surface!r}&ontology=mondo&exact=false&rows=1")
    print()

    normalized = _ols_search_mondo(surface)
    print("  Resulting NormalizedDisease:")
    for k, v in normalized.model_dump().items():
        print(f"    {k:12s} : {v!r}")
    if not normalized.mondo_id:
        print()
        print("  (No MONDO match. Either OLS returned no candidate or a network")
        print("   issue blocked the call; the resolver fails soft to surface-only.)")
    print()


def _inspect_verdict(rsid: str) -> None:
    rsid = rsid.strip().lower()
    _rule(f"VERDICT-LEVEL DISEASE NORMALIZATION for {rsid}")
    print("  Path:")
    print("    1. lookup_variant(rsid) -> VariantVerdict")
    print("    2. For each ClinVar condition, normalize_disease(cond)")
    print("    3. For each GWAS hit with a MONDO URI, normalize_disease(trait)")
    print("    4. Deduplicate by exact surface name across both sources")
    print()

    verdict = lookup_variant(rsid)
    print(f"  ClinVar conditions ({len(verdict.clinvar.conditions) if verdict.clinvar else 0}):")
    if verdict.clinvar and verdict.clinvar.conditions:
        for c in verdict.clinvar.conditions:
            print(f"    - {c!r}")
    else:
        print("    (none)")

    mondo_gwas = [a for a in verdict.gwas if a.trait and is_disease_uri(a.mapped_trait_uri)]
    print(f"  GWAS disease (MONDO) traits ({len(mondo_gwas)}):")
    if mondo_gwas:
        seen = set()
        for a in mondo_gwas:
            if a.trait in seen:
                continue
            seen.add(a.trait)
            print(f"    - {a.trait!r}  uri={a.mapped_trait_uri!r}")
    else:
        print("    (none)")
    print()

    print("  Running OLS lookups (one per unique surface)...")
    normalized = normalize_verdict_diseases(verdict)
    print()
    print(f"  Normalized diseases ({len(normalized)}):")
    for nd in normalized:
        marker = "[hit]" if nd.mondo_id else "[miss]"
        print(f"    {marker}")
        for k, v in nd.model_dump().items():
            print(f"      {k:10s} : {v!r}")
        print()


def inspect(text: str) -> None:
    _rule("INPUT")
    print(f"  Value: {text!r}")

    if _RSID_RE.match(text):
        print("  Type:  rsID -> verdict-level disease normalization")
        print()
        _inspect_verdict(text)
        return

    print("  Type:  surface form -> try gene first, then disease")
    print()

    gene = normalize_gene(text)
    _inspect_gene(text)

    if gene.symbol is None:
        # No gene match; also try as a disease.
        _inspect_disease(text)
    else:
        print("  (Gene match found; skipping disease lookup.")
        print("   To force a disease lookup on this string anyway, pass --disease.)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="normalize.inspect",
        description="Walk through the normalization stage step-by-step.",
    )
    parser.add_argument(
        "text",
        help="A gene symbol/alias, a disease name, or an rsID.",
    )
    parser.add_argument(
        "--disease",
        action="store_true",
        help="Force disease normalization (skip the gene attempt).",
    )
    parser.add_argument(
        "--gene",
        action="store_true",
        help="Force gene normalization (skip the disease attempt).",
    )
    args = parser.parse_args()

    if _RSID_RE.match(args.text):
        _rule("INPUT")
        print(f"  Value: {args.text!r}")
        print("  Type:  rsID -> verdict-level disease normalization")
        print()
        _inspect_verdict(args.text)
    elif args.disease:
        _rule("INPUT")
        print(f"  Value: {args.text!r}")
        print()
        _inspect_disease(args.text)
    elif args.gene:
        _rule("INPUT")
        print(f"  Value: {args.text!r}")
        print()
        _inspect_gene(args.text)
    else:
        inspect(args.text)


if __name__ == "__main__":
    main()
