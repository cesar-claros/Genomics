"""Reasoning layer: synthesize VariantVerdicts into grounded answers."""

from .models import Answer, Question
from .synthesize import synthesize

__all__ = ["synthesize", "Answer", "Question"]
