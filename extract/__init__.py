"""Entity extraction stage: text -> rsID + gene mentions -> verdicts."""

from .combine import PipelineResult, extract_and_lookup, extract_mentions
from .models import ExtractedMentions, GeneMention, VariantMention

__all__ = [
    "extract_mentions",
    "extract_and_lookup",
    "ExtractedMentions",
    "VariantMention",
    "GeneMention",
    "PipelineResult",
]
