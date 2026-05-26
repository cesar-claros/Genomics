"""
Load a VariantVerdict into the KG.

Each call is idempotent: re-running on the same rsID does not create
duplicate nodes or edges, thanks to MERGE and the uniqueness constraints
declared in schema.py.

Edges carry provenance: each ClinVar-derived edge keeps the review status,
significance string, and effect direction; each GWAS-derived edge keeps
the p-value, OR, risk allele, and source PMID. Down-pipeline queries can
filter by `source` (via edge type) or by stats fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lookup.heuristics import is_disease_uri
from lookup.models import VariantVerdict

from .client import Neo4jClient


@dataclass
class LoadStats:
    """Per-verdict counts of nodes and edges merged."""

    variants: int = 0
    genes: int = 0
    diseases: int = 0
    traits: int = 0
    studies: int = 0
    clinvar_edges: int = 0
    gwas_edges: int = 0
    cited_in_edges: int = 0

    def add(self, other: "LoadStats") -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))


def _upsert_variant_and_gene(tx, verdict: VariantVerdict) -> None:
    rsid = (verdict.rsid or verdict.query).strip().lower()
    tx.run(
        """
        MERGE (v:Variant {rsid: $rsid})
        SET v.overall_effect = $overall_effect,
            v.summary        = $summary
        """,
        rsid=rsid,
        overall_effect=verdict.overall_effect.value,
        summary=verdict.summary,
    )
    if verdict.gene:
        tx.run(
            """
            MERGE (v:Variant {rsid: $rsid})
            MERGE (g:Gene {symbol: $symbol})
            MERGE (v)-[:IN_GENE]->(g)
            """,
            rsid=rsid,
            symbol=verdict.gene,
        )


def _canonicalize_diseases(verdict: VariantVerdict) -> dict[str, dict[str, str | None]]:
    """Run MONDO normalization on every disease-flavored mention in the verdict.

    Returns a dict keyed by the surface name (the exact string we MERGE on)
    whose value carries the MONDO ID, canonical label, and IRI when
    available. Surfaces with no match map to an entry with all-None values.

    Fail-soft: if the OLS API is unreachable or the normalize package is
    not importable, returns {} and Disease nodes are MERGEd without
    canonical IDs (same as pre-integration behavior). The failure is
    logged to stderr rather than fully swallowed, so silent misses are
    easier to diagnose.
    """
    import sys

    try:
        from normalize.combine import normalize_verdict_diseases

        normalized = normalize_verdict_diseases(verdict)
    except Exception as e:  # noqa: BLE001 - fail soft, but log
        print(
            f"[kg.load] disease normalization skipped: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return {}

    return {
        nd.surface: {
            "mondo_id": nd.mondo_id,
            "canonical_label": nd.label,
            "iri": nd.iri,
        }
        for nd in normalized
    }


def _upsert_clinvar_conditions(
    tx,
    verdict: VariantVerdict,
    disease_norm: dict[str, dict[str, str | None]] | None = None,
) -> None:
    cv = verdict.clinvar
    if cv is None or not cv.conditions:
        return
    rsid = (verdict.rsid or verdict.query).strip().lower()

    rows = []
    for cond in cv.conditions:
        info = (disease_norm or {}).get(cond, {})
        rows.append(
            {
                "name": cond,
                "mondo_id": info.get("mondo_id"),
                "canonical_label": info.get("canonical_label"),
                "iri": info.get("iri"),
            }
        )

    tx.run(
        """
        UNWIND $rows AS row
        MERGE (v:Variant {rsid: $rsid})
        MERGE (d:Disease {name: row.name})
        SET d.mondo_id        = coalesce(row.mondo_id, d.mondo_id),
            d.canonical_label = coalesce(row.canonical_label, d.canonical_label),
            d.iri             = coalesce(row.iri, d.iri)
        MERGE (v)-[r:HAS_CLINVAR_CONDITION]->(d)
        SET r.review_stars  = $review_stars,
            r.significance  = $significance,
            r.review_status = $review_status,
            r.effect        = $effect
        """,
        rsid=rsid,
        rows=rows,
        review_stars=cv.review_stars,
        significance=cv.clinical_significance,
        review_status=cv.review_status,
        effect=cv.effect.value,
    )


def _upsert_gwas_associations(
    tx,
    verdict: VariantVerdict,
    disease_norm: dict[str, dict[str, str | None]] | None = None,
) -> None:
    if not verdict.gwas:
        return
    rsid = (verdict.rsid or verdict.query).strip().lower()

    # Split into disease (MONDO) hits and other (EFO measurement, OBA, etc.)
    # so each goes to a node with the right label. Disease rows get the
    # OLS-derived canonical fields when available.
    diseases = []
    traits = []
    for a in verdict.gwas:
        if a.trait is None:
            continue
        row = {
            "name": a.trait,
            "uri": a.mapped_trait_uri,
            "p_value": a.p_value,
            "odds_ratio": a.odds_ratio,
            "effect": a.effect.value,
            "risk_allele": a.risk_allele,
            "pmid": a.pubmed_id,
        }
        if is_disease_uri(a.mapped_trait_uri):
            info = (disease_norm or {}).get(a.trait, {})
            row["mondo_id"] = info.get("mondo_id")
            row["canonical_label"] = info.get("canonical_label")
            row["iri_canonical"] = info.get("iri")
            diseases.append(row)
        else:
            traits.append(row)

    if diseases:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (v:Variant {rsid: $rsid})
            MERGE (d:Disease {name: row.name})
            SET d.uri             = coalesce(row.uri, d.uri),
                d.mondo_id        = coalesce(row.mondo_id, d.mondo_id),
                d.canonical_label = coalesce(row.canonical_label, d.canonical_label),
                d.iri             = coalesce(row.iri_canonical, d.iri)
            MERGE (v)-[r:HAS_GWAS_ASSOCIATION {pmid: row.pmid}]->(d)
            SET r.p_value     = row.p_value,
                r.odds_ratio  = row.odds_ratio,
                r.effect      = row.effect,
                r.risk_allele = row.risk_allele
            """,
            rsid=rsid,
            rows=diseases,
        )
    if traits:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (v:Variant {rsid: $rsid})
            MERGE (t:Trait {name: row.name})
            SET t.uri = coalesce(row.uri, t.uri)
            MERGE (v)-[r:HAS_GWAS_ASSOCIATION {pmid: row.pmid}]->(t)
            SET r.p_value     = row.p_value,
                r.odds_ratio  = row.odds_ratio,
                r.effect      = row.effect,
                r.risk_allele = row.risk_allele
            """,
            rsid=rsid,
            rows=traits,
        )

    pmids = sorted({a.pubmed_id for a in verdict.gwas if a.pubmed_id})
    if pmids:
        tx.run(
            """
            UNWIND $pmids AS pmid
            MERGE (v:Variant {rsid: $rsid})
            MERGE (s:Study {pmid: pmid})
            MERGE (v)-[:CITED_IN]->(s)
            """,
            rsid=rsid,
            pmids=pmids,
        )


def _stats_for(verdict: VariantVerdict) -> LoadStats:
    stats = LoadStats(variants=1, genes=1 if verdict.gene else 0)
    if verdict.clinvar and verdict.clinvar.conditions:
        stats.diseases += len(set(verdict.clinvar.conditions))
        stats.clinvar_edges += len(set(verdict.clinvar.conditions))
    if verdict.gwas:
        named = [a for a in verdict.gwas if a.trait]
        disease_names = {a.trait for a in named if is_disease_uri(a.mapped_trait_uri)}
        trait_names = {a.trait for a in named if not is_disease_uri(a.mapped_trait_uri)}
        stats.diseases += len(disease_names)
        stats.traits += len(trait_names)
        # Edge cardinality after MERGE on (rsid, name, pmid):
        keyed = {(a.trait, a.pubmed_id) for a in named}
        stats.gwas_edges += len(keyed)
        pmids = {a.pubmed_id for a in verdict.gwas if a.pubmed_id}
        stats.studies += len(pmids)
        stats.cited_in_edges += len(pmids)
    return stats


def load_verdict(
    client: Neo4jClient,
    verdict: VariantVerdict,
    *,
    normalize_diseases: bool = True,
) -> LoadStats:
    """Upsert a single VariantVerdict into Neo4j. Returns counts of merged
    nodes and edges (these are upper bounds: MERGE will reuse existing
    nodes for repeated runs, so the actual delta may be smaller).

    If `normalize_diseases`, all disease/condition surfaces (ClinVar
    conditions and MONDO-flagged GWAS traits) are resolved against MONDO
    via OLS before loading, and the resulting `mondo_id`,
    `canonical_label`, and `iri` are stored as properties on the Disease
    nodes. Disable to skip the OLS network calls and load with name-only
    Disease nodes (same behavior as before the normalize-stage integration).
    """
    disease_norm = _canonicalize_diseases(verdict) if normalize_diseases else {}
    with client.session() as session:
        session.execute_write(_upsert_variant_and_gene, verdict)
        session.execute_write(_upsert_clinvar_conditions, verdict, disease_norm)
        session.execute_write(_upsert_gwas_associations, verdict, disease_norm)
    return _stats_for(verdict)


def load_verdicts(
    client: Neo4jClient,
    verdicts: list[VariantVerdict],
    *,
    normalize_diseases: bool = True,
) -> LoadStats:
    """Load multiple verdicts; accumulate stats."""
    total = LoadStats()
    for v in verdicts:
        total.add(load_verdict(client, v, normalize_diseases=normalize_diseases))
    return total
