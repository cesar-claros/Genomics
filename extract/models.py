"""
Typed records for the extraction stage.

Mentions are the input contract for the verdict layer: each VariantMention
carries enough information for lookup_variant (the normalized rsID) and
enough context for downstream traceability (the surface form + character
offsets, so the reasoning layer can quote the original text).

Kept deliberately small: no entity linking, no canonicalization beyond
lowercasing the rsID. Richer normalization (HGNC alias resolution, HGVS to
rsID, star alleles) is a separate layer that can be added without changing
this shape.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class VariantMention(BaseModel):
    """A single variant identifier found in source text."""

    text: str = Field(..., description="Raw surface form, e.g. 'RS429358'")
    rsid: str = Field(..., description="Normalized rsID, lowercase, e.g. 'rs429358'")
    start: int = Field(..., ge=0, description="Char offset of mention start in source text")
    end: int = Field(..., ge=0, description="Char offset of mention end (exclusive)")


class GeneMention(BaseModel):
    """A gene/gene-product mention emitted by the NER model."""

    text: str = Field(..., description="Surface form as it appears in text")
    symbol: str = Field(..., description="Best-effort canonical symbol (uppercased surface)")
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    label: str = Field(..., description="Underlying NER label, e.g. 'GENE_OR_GENE_PRODUCT'")


class ExtractedMentions(BaseModel):
    """All entity mentions found in a chunk of text. Mirrors the fail-soft
    contract of VariantVerdict: one extractor erroring still yields the other's
    results, with the failure recorded in `errors`."""

    text: str = Field(..., description="Original source text, kept for traceability")
    variants: list[VariantMention] = Field(default_factory=list)
    genes: list[GeneMention] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
