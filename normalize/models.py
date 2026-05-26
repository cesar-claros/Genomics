"""
Typed records for normalized entities.

Each Normalized* record always carries the original `surface` form, so
callers can recover the raw mention. Canonical fields are optional: if
the resolver couldn't find a match, the record is returned with
`surface` populated and the canonical fields left as None.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from extract.models import VariantMention


class NormalizedGene(BaseModel):
    """A gene mention resolved to canonical HGNC information."""

    surface: str = Field(..., description="Original surface form, e.g., 'Apo E'")
    symbol: str | None = Field(
        None, description="Approved HGNC symbol, e.g., 'APOE'"
    )
    hgnc_id: str | None = Field(
        None, description="HGNC accession, e.g., 'HGNC:613'"
    )
    entrez_id: str | None = Field(
        None, description="NCBI Entrez Gene ID, e.g., '348'"
    )
    ensembl_id: str | None = Field(
        None, description="Ensembl gene ID, e.g., 'ENSG00000130203'"
    )
    alias_type: str | None = Field(
        None,
        description="How the surface matched: 'symbol' (already canonical), "
                    "'alias', or 'previous'. None means no match.",
    )


class NormalizedDisease(BaseModel):
    """A disease/condition mention resolved to MONDO."""

    surface: str = Field(..., description="Original disease/condition name")
    label: str | None = Field(None, description="Canonical MONDO label")
    mondo_id: str | None = Field(
        None, description="MONDO accession, e.g., 'MONDO:0004975'"
    )
    iri: str | None = Field(None, description="Full ontology IRI")
    score: float | None = Field(
        None, description="OLS relevance score for the match (higher = better)"
    )


class NormalizedMentions(BaseModel):
    """Result of running normalize_mentions on an ExtractedMentions.

    Variants carry through unchanged from extract (rsIDs are already
    canonical). Genes get HGNC-resolved.
    """

    text: str = Field(..., description="Original source text")
    variants: list[VariantMention] = Field(default_factory=list)
    genes: list[NormalizedGene] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
