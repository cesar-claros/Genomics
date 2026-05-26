"""
High-level synthesis: VariantVerdict + question -> Answer.

This is the function the rest of the pipeline calls. It assembles the
prompt, invokes the configured backend, parses citations out of the
response, and cross-checks them against the verdict's actual PMIDs to
separate real citations from fabricated ones.

Fail-soft: backend errors are captured into Answer.errors rather than
raised, mirroring the lookup layer's contract so a failed synthesis still
produces a usable record (just with empty text).
"""

from __future__ import annotations

from lookup.models import VariantVerdict

from .backend import Backend, default_backend
from .models import Answer
from .prompt import build_prompt, extract_citations

_DEFAULT_QUESTION = (
    "Summarize what is known about this variant based on the evidence. "
    "What conditions or traits is it associated with, and how strong is "
    "the evidence?"
)


def _evidence_pmids(verdict: VariantVerdict) -> set[str]:
    """All PMIDs that appear in the verdict's evidence; the citation truth set."""
    return {g.pubmed_id for g in verdict.gwas if g.pubmed_id}


def synthesize(
    verdict: VariantVerdict,
    question: str | None = None,
    mention_context: str | None = None,
    backend: Backend | None = None,
) -> Answer:
    """Run the LLM synthesis over a single verdict."""
    backend = backend or default_backend()
    q = question or _DEFAULT_QUESTION

    system, user = build_prompt(verdict, q, mention_context=mention_context)

    try:
        text = backend.generate(system, user)
    except Exception as e:  # noqa: BLE001 - fail soft
        return Answer(
            rsid=verdict.rsid or verdict.query,
            question=q,
            text="",
            backend=backend.name,
            errors=[f"backend: {type(e).__name__}: {e}"],
        )

    cited = extract_citations(text)
    evidence = _evidence_pmids(verdict)
    real = [p for p in cited if p in evidence]
    fabricated = [p for p in cited if p not in evidence]

    return Answer(
        rsid=verdict.rsid or verdict.query,
        question=q,
        text=text,
        citations=real,
        fabricated_citations=fabricated,
        backend=backend.name,
    )
