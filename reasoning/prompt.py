"""
Prompt assembly for variant evidence synthesis.

The prompt is structured to:
  1. Frame the LLM as a synthesizer, not a recall agent.
  2. Lay out evidence in a predictable schema so the model can quote it.
  3. Enforce citation discipline (PMIDs in a recognizable format).

The "do not recall from memory" instruction is load-bearing: it is the
difference between RAG and hallucinated genomics. Phrased strongly and
restated where the model is most likely to slip (when evidence is thin).
"""

from __future__ import annotations

import re

from lookup.models import VariantVerdict

_SYSTEM = """You are a biomedical evidence synthesizer.

You will receive curated evidence about a genomic variant from ClinVar
and the GWAS Catalog, plus a user question. Answer the question USING
ONLY THE PROVIDED EVIDENCE.

Grounding rules:
- Do NOT recall facts from memory. If the evidence does not address a
  claim, say so explicitly. Do not fill in details that are not in the
  evidence block, even if you "know" them from training.
- Cite each substantive claim inline using the exact format [PMID:12345].
  Only use PMIDs that appear in the evidence. Do not invent PMIDs.
- Avoid clinical recommendations. The evidence is population-level and
  research-grade only.

Evidence weighting:
- ClinVar review stars are confidence signals: 3-4 stars are well
  established, 2 stars are reasonable, 0-1 stars are weak.
- GWAS p-values gauge association strength.
- Disease and clinical-outcome traits (e.g., Alzheimer disease, Lewy
  body dementia, cancers, cardiovascular events) take precedence over
  biomarker, measurement, and laboratory-value traits (e.g., protein
  levels, neuroimaging measurements, fatty acid percentages,
  phospholipid amounts). Lead with the disease associations.

Output format:
- Write 3-5 sentences of flowing prose. No bulleted lists, no numbered
  lists, no tables, no trait-by-trait enumerations.
- Cite inline within the sentences, not as a trailing bibliography.
- Open with the strongest disease association if one exists. Mention
  biomarker associations only briefly, and only if they materially add
  to the picture.
"""


def _format_clinvar(verdict: VariantVerdict) -> str:
    cv = verdict.clinvar
    if cv is None or (cv.effect.value == "unknown" and not cv.clinical_significance):
        return "ClinVar: no record found for this variant."

    lines = ["ClinVar:"]
    if cv.clinical_significance:
        lines.append(f"  Clinical significance: {cv.clinical_significance}")
    if cv.review_status:
        stars = f" ({cv.review_stars} stars)" if cv.review_stars is not None else ""
        lines.append(f"  Review status: {cv.review_status}{stars}")
    if cv.conditions:
        lines.append(f"  Conditions: {'; '.join(cv.conditions)}")
    lines.append(f"  Effect direction: {cv.effect.value}")
    return "\n".join(lines)


def _format_gwas(verdict: VariantVerdict, max_assocs: int = 10) -> str:
    if not verdict.gwas:
        return "GWAS Catalog: no associations found."
    # Most significant first.
    assocs = sorted(
        verdict.gwas,
        key=lambda g: g.p_value if g.p_value is not None else float("inf"),
    )[:max_assocs]
    lines = [f"GWAS Catalog (top {len(assocs)} by p-value):"]
    for a in assocs:
        bits = [f"Trait: {a.trait or '(unspecified)'}"]
        if a.p_value is not None:
            bits.append(f"p={a.p_value:.1e}")
        if a.odds_ratio is not None:
            bits.append(f"OR={a.odds_ratio:.3f}")
        if a.effect.value != "unknown":
            bits.append(f"effect={a.effect.value}")
        if a.risk_allele:
            bits.append(f"risk_allele={a.risk_allele}")
        if a.pubmed_id:
            bits.append(f"PMID:{a.pubmed_id}")
        lines.append("  - " + "; ".join(bits))
    return "\n".join(lines)


def build_prompt(
    verdict: VariantVerdict,
    question: str,
    mention_context: str | None = None,
) -> tuple[str, str]:
    """Return (system_message, user_message) for the chat-style LLM."""
    parts = [f"Variant: {verdict.rsid or verdict.query}"]
    if verdict.gene:
        parts.append(f"Gene: {verdict.gene}")
    if mention_context:
        parts.append(f"Mention context: \"{mention_context}\"")
    parts.append("")
    parts.append(_format_clinvar(verdict))
    parts.append("")
    parts.append(_format_gwas(verdict))
    parts.append("")
    parts.append(f"Question: {question}")

    return _SYSTEM.strip(), "\n".join(parts)


_PMID_RE = re.compile(r"PMID[:\s]?(\d+)", re.IGNORECASE)


def extract_citations(text: str) -> list[str]:
    """Pull PMIDs out of the model's response in source order, deduped."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _PMID_RE.finditer(text):
        pmid = m.group(1)
        if pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out
