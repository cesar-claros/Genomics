"""
Classification heuristics for GWAS traits and trait URIs.

These are simple substring rules used in multiple places in the pipeline:
  - lookup/combine.py picks a single summary trait from a VariantVerdict and
    needs to know which GWAS hits are disease-flavored vs biomarker-flavored.
  - reasoning/prompt.py orders evidence for the LLM and needs the same
    distinction (so disease associations don't get buried behind pQTL/eQTL
    hits with much lower p-values).

Keep both rules here so they cannot drift between consumers.
"""

from __future__ import annotations

# Substrings in a trait name that mark it as a biomarker / quantitative
# measurement rather than a disease. Covers the long tail of EFO/OBA
# measurement terms ("protein measurement", "phospholipid amount",
# "...percentage", etc.).
MEASUREMENT_TOKENS = ("measurement", "amount", "level", "concentration", "percentage")


def is_disease_uri(uri: str | None) -> bool:
    """MONDO IDs denote diseases by construction; the strongest disease signal."""
    return bool(uri) and "MONDO_" in uri


def is_measurement_trait(trait: str | None) -> bool:
    """Cheap substring heuristic for 'this is a biomarker / quant trait'."""
    if not trait:
        return False
    t = trait.lower()
    return any(tok in t for tok in MEASUREMENT_TOKENS)
