"""
Typed records for the reasoning stage.

Answer is what the synthesizer returns: the LLM's narrative text, plus
machine-extracted citations split into those that match verdict evidence
(`citations`) and those that don't (`fabricated_citations`). Keeping these
separate makes the hallucination check observable rather than something a
caller has to recompute.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Question(BaseModel):
    """A user question paired with variant context."""

    rsid: str = Field(..., description="The variant the question is about")
    text: str = Field(..., description="Free-text user question")
    mention_context: str | None = Field(
        None,
        description="Surrounding text where the rsID was mentioned, if available",
    )


class Answer(BaseModel):
    """A grounded synthesis with citations checked against the evidence."""

    rsid: str
    question: str
    text: str = Field(..., description="The LLM's synthesized answer")
    citations: list[str] = Field(
        default_factory=list,
        description="PMIDs cited by the LLM that ARE in the verdict evidence",
    )
    fabricated_citations: list[str] = Field(
        default_factory=list,
        description="PMIDs cited by the LLM that are NOT in the verdict evidence "
                    "(likely hallucinations; treat with suspicion)",
    )
    backend: str = Field(..., description="Which model produced the answer")
    errors: list[str] = Field(default_factory=list)

    disclaimer: str = Field(
        default=(
            "Research/educational synthesis only. Population-level, probabilistic "
            "associations from public databases. Not a diagnosis or medical advice."
        ),
        description="Carried through; do not drop in downstream output.",
    )
