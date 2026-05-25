"""Variant -> disease verdict lookup (ClinVar + GWAS Catalog)."""

from .combine import lookup_variant
from .models import (
    ClinVarRecord,
    EffectDirection,
    GwasAssociation,
    VariantVerdict,
)

__all__ = [
    "lookup_variant",
    "VariantVerdict",
    "ClinVarRecord",
    "GwasAssociation",
    "EffectDirection",
]
