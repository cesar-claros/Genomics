"""
Orchestration: normalize a batch of mentions or all diseases in a verdict.

Two entry points:

  - `normalize_mentions(ExtractedMentions)`: takes the NER output (with
    surface gene names and raw rsIDs) and returns NormalizedMentions
    where gene surface forms are resolved to HGNC. rsIDs pass through
    unchanged because they were already canonicalized at extraction.

  - `normalize_verdict_diseases(VariantVerdict)`: walks the verdict's
    ClinVar conditions and GWAS-Catalog disease (MONDO-URI) traits,
    runs each through MONDO normalization, and returns the list of
    NormalizedDisease records. Useful for cross-source disease linking
    (e.g., recognizing that ClinVar's free-text "Breast-ovarian cancer
    susceptibility 1" and GWAS's "breast cancer" map to the same MONDO).

Both functions fail-soft: per-mention exceptions are captured in the
result's `errors` field or returned as stub records, never raised out.
"""

from __future__ import annotations

from extract.models import ExtractedMentions
from lookup.heuristics import is_disease_uri
from lookup.models import VariantVerdict

from .diseases import normalize_disease
from .genes import normalize_gene
from .models import NormalizedDisease, NormalizedMentions


def normalize_mentions(mentions: ExtractedMentions) -> NormalizedMentions:
    """Resolve gene mentions to HGNC; pass variants through unchanged."""
    result = NormalizedMentions(text=mentions.text, variants=list(mentions.variants))
    for g in mentions.genes:
        try:
            result.genes.append(normalize_gene(g.text))
        except Exception as e:  # noqa: BLE001 - fail soft per mention
            result.errors.append(f"gene({g.text!r}): {type(e).__name__}: {e}")
    return result


def normalize_verdict_diseases(verdict: VariantVerdict) -> list[NormalizedDisease]:
    """Resolve all disease/condition names in a verdict to MONDO.

    Includes:
      - Every ClinVar condition (ClinVar's mandate is disease curation,
        so every condition string is a disease by construction).
      - Every GWAS trait whose `mapped_trait_uri` is a MONDO URI (the
        same disease-vs-biomarker partition the rest of the pipeline
        uses, via `lookup.heuristics.is_disease_uri`).

    GWAS traits with non-MONDO URIs (EFO measurement, OBA, etc.) are
    biomarker traits, not diseases, and are skipped here. Normalizing
    biomarker traits to EFO IDs is a separate concern that would live
    in its own resolver.

    The same surface string is normalized at most once (deduplication
    by exact name), so a disease that appears in both ClinVar and GWAS
    produces a single NormalizedDisease record.
    """
    seen: set[str] = set()
    out: list[NormalizedDisease] = []

    if verdict.clinvar and verdict.clinvar.conditions:
        for cond in verdict.clinvar.conditions:
            if cond in seen or not cond:
                continue
            seen.add(cond)
            out.append(normalize_disease(cond))

    for assoc in verdict.gwas:
        trait = assoc.trait
        if not trait or trait in seen:
            continue
        if not is_disease_uri(assoc.mapped_trait_uri):
            continue
        seen.add(trait)
        out.append(normalize_disease(trait))

    return out
