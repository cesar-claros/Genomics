"""
Inspection tool for the KG stage.

Given an rsID or free text, runs the lookup, then prints every node and
edge the loader WOULD MERGE into Neo4j, with provenance, plus the
underlying Cypher pattern. Pure dry-run: no Neo4j connection is opened,
no environment variables for NEO4J_* are required.

Pedagogical, not for production. To actually load, use `python -m kg`.

Usage:
  python -m kg.inspect rs429358
  python -m kg.inspect "We genotyped APOE rs429358 in patients with AD."
"""

from __future__ import annotations

# Load code/.env so the lookup stage's NCBI env vars are populated.
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import argparse  # noqa: E402
import re  # noqa: E402

from extract.combine import extract_and_lookup  # noqa: E402
from lookup.combine import lookup_variant  # noqa: E402
from lookup.heuristics import is_disease_uri  # noqa: E402
from lookup.models import VariantVerdict  # noqa: E402

from .load import LoadStats, _stats_for  # noqa: E402
from .schema import NodeLabel, RelType  # noqa: E402

_RSID_ONLY_RE = re.compile(r"^\s*rs\d+\s*$", re.IGNORECASE)


def _rule(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def _inspect_verdict(verdict: VariantVerdict) -> LoadStats:
    rsid = (verdict.rsid or verdict.query).strip().lower()

    _rule(f"VERDICT: {rsid}")
    print(f"  overall_effect : {verdict.overall_effect.value}")
    print(f"  summary        : {verdict.summary!r}")
    print(f"  gene           : {verdict.gene!r}")
    print(f"  errors         : {verdict.errors}")
    print()

    # --- Nodes ---
    _rule(f"NODES that would be MERGEd for {rsid}")

    print(f"  [{NodeLabel.VARIANT.value}]  rsid={rsid!r}, "
          f"overall_effect={verdict.overall_effect.value!r}, "
          f"summary={verdict.summary!r}")
    print()

    if verdict.gene:
        print(f"  [{NodeLabel.GENE.value}]     symbol={verdict.gene!r}")
        print()

    cv = verdict.clinvar
    if cv and cv.conditions:
        print(f"  [{NodeLabel.DISEASE.value}]  (from ClinVar conditions)")
        for c in cv.conditions:
            print(f"    name={c!r}")
        print()

    if verdict.gwas:
        diseases = []
        traits = []
        for a in verdict.gwas:
            if a.trait is None:
                continue
            (diseases if is_disease_uri(a.mapped_trait_uri) else traits).append(a)

        if diseases:
            print(f"  [{NodeLabel.DISEASE.value}]  (from GWAS hits with MONDO URI)")
            seen = set()
            for a in diseases:
                if a.trait in seen:
                    continue
                seen.add(a.trait)
                print(f"    name={a.trait!r}  uri={a.mapped_trait_uri!r}")
            print()

        if traits:
            print(f"  [{NodeLabel.TRAIT.value}]    (from GWAS hits without MONDO URI)")
            seen = set()
            for a in traits:
                if a.trait in seen:
                    continue
                seen.add(a.trait)
                print(f"    name={a.trait!r}  uri={a.mapped_trait_uri!r}")
            print()

        pmids = sorted({a.pubmed_id for a in verdict.gwas if a.pubmed_id})
        if pmids:
            print(f"  [{NodeLabel.STUDY.value}]    (from GWAS PMIDs)")
            for p in pmids:
                print(f"    pmid={p!r}")
            print()

    # --- Edges ---
    _rule(f"EDGES that would be MERGEd for {rsid}")

    if verdict.gene:
        print(
            f"  ({NodeLabel.VARIANT.value} {{rsid={rsid!r}}})"
            f"  -[:{RelType.IN_GENE.value}]->"
            f"  ({NodeLabel.GENE.value} {{symbol={verdict.gene!r}}})"
        )
        print()

    if cv and cv.conditions:
        for c in cv.conditions:
            print(
                f"  ({NodeLabel.VARIANT.value})"
                f"  -[:{RelType.HAS_CLINVAR_CONDITION.value} "
                f"{{review_stars={cv.review_stars!r}, "
                f"significance={cv.clinical_significance!r}, "
                f"effect={cv.effect.value!r}}}]->"
                f"  ({NodeLabel.DISEASE.value} {{name={c!r}}})"
            )
        print()

    if verdict.gwas:
        keyed = set()
        for a in verdict.gwas:
            if not a.trait:
                continue
            label = (
                NodeLabel.DISEASE.value
                if is_disease_uri(a.mapped_trait_uri)
                else NodeLabel.TRAIT.value
            )
            edge_key = (a.trait, a.pubmed_id)
            if edge_key in keyed:
                continue
            keyed.add(edge_key)
            print(
                f"  ({NodeLabel.VARIANT.value})"
                f"  -[:{RelType.HAS_GWAS_ASSOCIATION.value} "
                f"{{p={a.p_value!s}, OR={a.odds_ratio!s}, "
                f"effect={a.effect.value!r}, "
                f"risk_allele={a.risk_allele!r}, "
                f"pmid={a.pubmed_id!r}}}]->"
                f"  ({label} {{name={a.trait!r}}})"
            )

        pmids = sorted({a.pubmed_id for a in verdict.gwas if a.pubmed_id})
        if pmids:
            print()
            for p in pmids:
                print(
                    f"  ({NodeLabel.VARIANT.value})"
                    f"  -[:{RelType.CITED_IN.value}]->"
                    f"  ({NodeLabel.STUDY.value} {{pmid={p!r}}})"
                )
        print()

    stats = _stats_for(verdict)
    return stats


def inspect(text: str) -> None:
    _rule("INPUT")
    if _RSID_ONLY_RE.match(text):
        print(f"  Type   : direct rsID")
        print(f"  Value  : {text.strip().lower()!r}")
        print("  Path   : lookup_variant (skips NER)")
    else:
        print(f"  Type   : free text")
        print(f"  Value  : {text!r}")
        print("  Path   : extract_and_lookup (rsID regex + scispaCy gene NER)")
    print()

    if _RSID_ONLY_RE.match(text):
        verdicts = [lookup_variant(text.strip().lower())]
    else:
        pipeline = extract_and_lookup(text)
        _rule("EXTRACT STAGE OUTPUT")
        print(f"  Variant mentions: {len(pipeline.mentions.variants)}")
        for v in pipeline.mentions.variants:
            print(f"    - text={v.text!r}  rsid={v.rsid!r}  span={v.start}-{v.end}")
        print(f"  Gene mentions:    {len(pipeline.mentions.genes)}")
        for g in pipeline.mentions.genes:
            print(f"    - text={g.text!r}  symbol={g.symbol!r}  span={g.start}-{g.end}")
        print()
        verdicts = pipeline.verdicts

    total = LoadStats()
    for v in verdicts:
        s = _inspect_verdict(v)
        for f in s.__dataclass_fields__:
            setattr(total, f, getattr(total, f) + getattr(s, f))

    _rule("AGGREGATE STATS (what `python -m kg` would report as `merged`)")
    for f in total.__dataclass_fields__:
        print(f"  {f:20s} : {getattr(total, f)}")
    print()

    _rule("UNDERLYING CYPHER PATTERN")
    print("""
  // Variant + Gene (gene only if verdict.gene is set)
  MERGE (v:Variant {rsid: $rsid})
  SET   v.overall_effect = $overall_effect, v.summary = $summary
  // ... then, optionally:
  MERGE (g:Gene {symbol: $gene})
  MERGE (v)-[:IN_GENE]->(g)

  // ClinVar conditions (one edge per condition)
  UNWIND $conditions AS cond
  MERGE (v:Variant {rsid: $rsid})
  MERGE (d:Disease {name: cond})
  MERGE (v)-[r:HAS_CLINVAR_CONDITION]->(d)
  SET r.review_stars = $review_stars,
      r.significance = $significance,
      r.effect       = $effect

  // GWAS associations (one edge per (trait, pmid); Disease for MONDO URIs,
  // Trait otherwise)
  UNWIND $rows AS row
  MERGE (v:Variant {rsid: $rsid})
  MERGE (n:Disease {name: row.name})       // or :Trait, per is_disease_uri()
  SET n.uri = coalesce(row.uri, n.uri)
  MERGE (v)-[r:HAS_GWAS_ASSOCIATION {pmid: row.pmid}]->(n)
  SET r.p_value = row.p_value, r.odds_ratio = row.odds_ratio,
      r.effect  = row.effect,  r.risk_allele = row.risk_allele

  // Study citations (one per unique PMID)
  UNWIND $pmids AS pmid
  MERGE (v:Variant {rsid: $rsid})
  MERGE (s:Study {pmid: pmid})
  MERGE (v)-[:CITED_IN]->(s)
""")
    print("  All operations use MERGE (idempotent upsert), so re-running on the")
    print("  same rsID does not duplicate nodes or edges.")
    print()

    _rule("TO ACTUALLY LOAD INTO NEO4J")
    print("  python -m kg <same input>")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kg.inspect",
        description="Dry-run inspection of what `python -m kg` would load.",
    )
    parser.add_argument(
        "input",
        help="An rsID or free-text passage containing rsIDs / gene mentions.",
    )
    args = parser.parse_args()
    inspect(args.input)


if __name__ == "__main__":
    main()
