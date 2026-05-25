"""
Combine rsID + gene extraction, then optionally chain into the verdict layer.

Mirrors the fail-soft contract of lookup.combine: a single extractor erroring
still yields the other's results, with the failure recorded in `errors`.

The chained convenience function `extract_and_lookup` associates each variant
mention with the nearest gene mention (by character offset) and passes the
gene symbol through to lookup_variant. This is a deliberately simple proxy
for real coreference resolution; replacing it with sentence-level alignment
or a proper linker is the next iteration's job.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lookup.combine import lookup_variant
from lookup.models import VariantVerdict

from .genes import extract_genes
from .models import ExtractedMentions, GeneMention, VariantMention
from .rsid import extract_rsids


class PipelineResult(BaseModel):
    """Bundle of what was found and what each variant looked up to."""

    mentions: ExtractedMentions
    verdicts: list[VariantVerdict] = Field(default_factory=list)


def extract_mentions(text: str) -> ExtractedMentions:
    """Run all extractors on `text`. Per-extractor errors are captured, not raised."""
    result = ExtractedMentions(text=text)

    try:
        result.variants = extract_rsids(text)
    except Exception as e:  # noqa: BLE001 - fail soft, record and continue
        result.errors.append(f"rsid: {type(e).__name__}: {e}")

    try:
        result.genes = extract_genes(text)
    except Exception as e:  # noqa: BLE001 - fail soft
        result.errors.append(f"genes: {type(e).__name__}: {e}")

    return result


def _nearest_gene(
    variant: VariantMention,
    genes: list[GeneMention],
    max_distance: int,
) -> str | None:
    """Symbol of the gene mention closest to `variant` by character offset,
    or None if no gene mention falls within `max_distance` characters."""
    v_mid = (variant.start + variant.end) // 2
    best_symbol: str | None = None
    best_distance = max_distance + 1
    for g in genes:
        g_mid = (g.start + g.end) // 2
        d = abs(v_mid - g_mid)
        if d <= max_distance and d < best_distance:
            best_distance = d
            best_symbol = g.symbol
    return best_symbol


def extract_and_lookup(
    text: str,
    *,
    gene_proximity: int = 120,
    dedupe: bool = True,
    use_clinvar: bool = True,
    use_gwas: bool = True,
) -> PipelineResult:
    """
    Run extraction over `text`, then look up each variant mention.

    Each variant is paired with the nearest gene mention within
    `gene_proximity` characters (~ one sentence), whose symbol is passed as
    context to lookup_variant. If `dedupe`, repeated rsIDs in the same text
    are looked up once.
    """
    mentions = extract_mentions(text)

    seen: set[str] = set()
    verdicts: list[VariantVerdict] = []
    for v in mentions.variants:
        if dedupe and v.rsid in seen:
            continue
        seen.add(v.rsid)
        gene = _nearest_gene(v, mentions.genes, gene_proximity)
        verdicts.append(
            lookup_variant(
                v.rsid,
                use_clinvar=use_clinvar,
                use_gwas=use_gwas,
                gene=gene,
            )
        )

    return PipelineResult(mentions=mentions, verdicts=verdicts)
