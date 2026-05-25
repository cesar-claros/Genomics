"""
Typed records for variant -> disease verdicts.

These are the contract between the lookup layer and everything downstream
(the reasoning/synthesis layer and the Neo4j loader). Keeping them typed means
the rest of the pipeline gets autocomplete + validation instead of guessing at
dict keys.

IMPORTANT FRAMING: these records describe POPULATION-LEVEL, PROBABILISTIC
associations from public databases. They are research/educational information,
not a diagnosis or personal medical advice. The `disclaimer` field is carried
through deliberately so downstream output can't silently drop it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EffectDirection(str, Enum):
    """Which way a variant pushes disease risk."""

    RISK = "risk_increasing"        # pathogenic / risk allele
    PROTECTIVE = "protective"       # decreases risk
    BENIGN = "benign"               # no/neutral effect
    UNCERTAIN = "uncertain"         # VUS or conflicting evidence
    UNKNOWN = "unknown"             # not found in this source


class ClinVarRecord(BaseModel):
    """One variant's clinical-significance picture from ClinVar."""

    rsid: str | None = Field(None, description="dbSNP rsID, e.g. rs429358")
    clinvar_variation_id: str | None = Field(
        None, description="ClinVar VariationID (stable ClinVar key)"
    )
    clinical_significance: str | None = Field(
        None, description="Raw ClinVar significance, e.g. 'Pathogenic', 'Benign'"
    )
    review_status: str | None = Field(
        None, description="ClinVar review status (drives the star rating)"
    )
    review_stars: int | None = Field(
        None, ge=0, le=4, description="0-4 star confidence derived from review_status"
    )
    conditions: list[str] = Field(
        default_factory=list, description="Associated condition/disease names"
    )
    effect: EffectDirection = Field(
        EffectDirection.UNKNOWN, description="Normalized effect direction"
    )


class GwasAssociation(BaseModel):
    """One variant-trait association from the GWAS Catalog."""

    rsid: str | None = None
    trait: str | None = Field(None, description="Reported trait/disease")
    mapped_trait_uri: str | None = Field(
        None, description="EFO ontology URI for the trait, when available"
    )
    p_value: float | None = Field(None, description="Association p-value")
    odds_ratio: float | None = Field(
        None, description="Effect size (OR or beta) if reported"
    )
    risk_allele: str | None = Field(
        None, description="The allele the effect is reported for"
    )
    effect: EffectDirection = Field(EffectDirection.UNKNOWN)
    pubmed_id: str | None = Field(None, description="Source study PMID, for citation")


class VariantVerdict(BaseModel):
    """
    The combined, pipeline-facing record for a single variant. This is what the
    lookup client returns and what the reasoning layer consumes.
    """

    query: str = Field(..., description="The variant as the user asked for it")
    rsid: str | None = Field(None, description="Resolved/normalized rsID")
    gene: str | None = Field(None, description="Associated gene symbol, if known")

    clinvar: ClinVarRecord | None = None
    gwas: list[GwasAssociation] = Field(default_factory=list)

    # A coarse roll-up across sources for quick filtering/display. The nuanced
    # synthesis (handling conflicts between sources) is the reasoning layer's job.
    overall_effect: EffectDirection = Field(EffectDirection.UNKNOWN)
    summary: str | None = Field(
        None, description="One-line plain-language roll-up (no clinical claims)"
    )

    sources_queried: list[str] = Field(default_factory=list)
    errors: list[str] = Field(
        default_factory=list,
        description="Per-source errors (so a partial result is still usable)",
    )

    disclaimer: str = Field(
        default=(
            "Research/educational information only. These are population-level, "
            "probabilistic associations from public databases — not a diagnosis "
            "or personal medical advice. Consult a qualified clinician or genetic "
            "counselor for personal interpretation."
        ),
        description="Carried through to all downstream output; do not drop.",
    )
