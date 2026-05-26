"""Normalization / entity-linking stage.

Maps surface-form mentions to canonical identifiers:
  - Genes: surface symbol -> HGNC approved symbol + HGNC ID.
  - Diseases: free-text name -> MONDO ID + canonical label.
  - Variants: rsIDs are already canonical from the extract stage;
    HGVS / star alleles deferred until HGVS extraction lands.

Standalone for v1: not yet wired into extract_and_lookup or kg/load.
Call explicitly to enrich an ExtractedMentions or a VariantVerdict.
"""

from .combine import normalize_mentions, normalize_verdict_diseases
from .diseases import normalize_disease
from .genes import normalize_gene
from .models import NormalizedDisease, NormalizedGene, NormalizedMentions

__all__ = [
    "normalize_gene",
    "normalize_disease",
    "normalize_mentions",
    "normalize_verdict_diseases",
    "NormalizedGene",
    "NormalizedDisease",
    "NormalizedMentions",
]
