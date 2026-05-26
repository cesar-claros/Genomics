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


def _upsert_clinvar_conditions(tx, verdict: VariantVerdict) -> None:
    cv = verdict.clinvar
    if cv is None or not cv.conditions:
        return
    rsid = (verdict.rsid or verdict.query).strip().lower()
    tx.run(
        """
        UNWIND $conditions AS cond
        MERGE (v:Variant {rsid: $rsid})
        MERGE (d:Disease {name: cond})
        MERGE (v)-[r:HAS_CLINVAR_CONDITION]->(d)
        SET r.review_stars = $review_stars,
            r.significance = $significance,
            r.review_status = $review_status,
            r.effect       = $effect
        """,
        rsid=rsid,
        conditions=cv.conditions,
        review_stars=cv.review_stars,
        significance=cv.clinical_significance,
        review_status=cv.review_status,
        effect=cv.effect.value,
    )


def _upsert_gwas_associations(tx, verdict: VariantVerdict) -> None:
    if not verdict.gwas:
        return
    rsid = (verdict.rsid or verdict.query).strip().lower()

    # Split into disease (MONDO) hits and other (EFO measurement, OBA, etc.)
    # so each goes to a node with the right label.
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
        (diseases if is_disease_uri(a.mapped_trait_uri) else traits).append(row)

    if diseases:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (v:Variant {rsid: $rsid})
            MERGE (d:Disease {name: row.name})
            SET d.uri = coalesce(row.uri, d.uri)
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


def load_verdict(client: Neo4jClient, verdict: VariantVerdict) -> LoadStats:
    """Upsert a single VariantVerdict into Neo4j. Returns counts of merged
    nodes and edges (these are upper bounds: MERGE will reuse existing
    nodes for repeated runs, so the actual delta may be smaller)."""
    with client.session() as session:
        session.execute_write(_upsert_variant_and_gene, verdict)
        session.execute_write(_upsert_clinvar_conditions, verdict)
        session.execute_write(_upsert_gwas_associations, verdict)
    return _stats_for(verdict)


def load_verdicts(
    client: Neo4jClient,
    verdicts: list[VariantVerdict],
) -> LoadStats:
    """Load multiple verdicts; accumulate stats."""
    total = LoadStats()
    for v in verdicts:
        total.add(load_verdict(client, v))
    return total
