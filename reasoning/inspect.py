"""
Inspection tool for the reasoning stage.

Given an rsID, runs the lookup (so the LLM has something to synthesize)
then walks through the synthesis: the assembled (system, user) prompt
verbatim, the configured backend, the LLM's raw response, the citation
regex matches, and the real-vs-fabricated PMID split.

Pedagogical, not for production. The LLM call is real; the rest is
just printing.

Usage:
  python -m reasoning.inspect rs429358
  python -m reasoning.inspect rs429358 --question "Is this protective in any context?"
"""

from __future__ import annotations

# Load code/.env BEFORE other imports so REASONING_MODEL / REASONING_DTYPE
# and lookup env vars are visible.
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import argparse  # noqa: E402
import time  # noqa: E402

from lookup.combine import lookup_variant  # noqa: E402

from .backend import default_backend  # noqa: E402
from .prompt import build_prompt, extract_citations  # noqa: E402

_DEFAULT_QUESTION = (
    "Summarize what is known about this variant based on the evidence. "
    "What conditions or traits is it associated with, and how strong is "
    "the evidence?"
)


def _rule(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def inspect(rsid: str, question: str) -> None:
    _rule("INPUT")
    print(f"  rsID:     {rsid}")
    print(f"  question: {question}")
    print()

    _rule("STAGE 1 - LOOKUP (compressed)")
    print("  Calling lookup_variant(rsid) ...")
    t0 = time.perf_counter()
    verdict = lookup_variant(rsid)
    dt = time.perf_counter() - t0
    print(f"  Done in {dt:.2f}s.")
    print(f"    overall_effect : {verdict.overall_effect.value}")
    print(f"    summary        : {verdict.summary!r}")
    print(f"    gene           : {verdict.gene!r}")
    if verdict.clinvar:
        print(
            f"    ClinVar        : sig={verdict.clinvar.clinical_significance!r}, "
            f"stars={verdict.clinvar.review_stars}, "
            f"conditions={len(verdict.clinvar.conditions)}"
        )
    print(f"    GWAS assocs    : {len(verdict.gwas)}")
    evidence_pmids = sorted({g.pubmed_id for g in verdict.gwas if g.pubmed_id})
    print(f"    evidence PMIDs : {len(evidence_pmids)} unique")
    print(f"    errors         : {verdict.errors}")
    print()

    _rule("STAGE 2 - PROMPT ASSEMBLY (build_prompt)")
    system, user = build_prompt(verdict, question, mention_context=None)
    print("--- SYSTEM MESSAGE ---")
    print(system)
    print()
    print("--- USER MESSAGE ---")
    print(user)
    print()

    _rule("STAGE 3 - BACKEND")
    backend = default_backend()
    print(f"  Backend       : {backend.name}")
    print(f"  Model (REASONING_MODEL) : {backend.model_name}")
    print(f"  dtype (REASONING_DTYPE) : {backend.dtype}")
    print()
    print("  Note: lazy-load. First call to backend.generate() will:")
    print("    1. AutoTokenizer.from_pretrained(model_name)")
    print("    2. AutoModelForCausalLM.from_pretrained(model_name, dtype=..., device_map='auto')")
    print("    3. tokenizer.apply_chat_template -> prompt string")
    print("    4. model.generate(do_sample=False, max_new_tokens=1024)")
    print("    5. Decode only the newly-generated tokens (skip prompt echo).")
    print()

    _rule("STAGE 4 - LLM RAW OUTPUT")
    print("  Generating ...")
    t0 = time.perf_counter()
    raw = backend.generate(system, user)
    dt = time.perf_counter() - t0
    print(f"  Done in {dt:.2f}s.")
    print()
    print(raw)
    print()

    _rule("STAGE 5 - CITATION REGEX MATCH")
    print(r"  Pattern: PMID[:\s]?(\d+)  (case-insensitive)")
    cited = extract_citations(raw)
    print(f"  PMIDs found in response: {cited}")
    print()

    _rule("STAGE 6 - HALLUCINATION CHECK (PMID cross-validation)")
    print(f"  Evidence PMIDs (from verdict.gwas[*].pubmed_id): {evidence_pmids}")
    print(f"  Cited PMIDs                                    : {cited}")
    real = [p for p in cited if p in set(evidence_pmids)]
    fabricated = [p for p in cited if p not in set(evidence_pmids)]
    print()
    print(f"  Real citations (Answer.citations)              : {real}")
    print(f"  Fabricated (Answer.fabricated_citations)       : {fabricated}")
    if fabricated:
        print()
        print("  WARNING: The LLM cited PMIDs that are NOT in the verdict's evidence.")
        print("  These are flagged in Answer.fabricated_citations so any downstream")
        print("  consumer respecting the contract knows they are unreliable.")
    print()

    _rule("FINAL ANSWER (synthesize would return)")
    print(f"  rsid                  : {rsid}")
    print(f"  text                  : (see Stage 4 above)")
    print(f"  citations             : {real}")
    print(f"  fabricated_citations  : {fabricated}")
    print(f"  backend               : {backend.name}")
    print(f"  errors                : []")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reasoning.inspect",
        description="Walk through the reasoning stage step-by-step for one rsID.",
    )
    parser.add_argument("rsid", help="A dbSNP rsID, e.g. rs429358.")
    parser.add_argument(
        "--question",
        default=_DEFAULT_QUESTION,
        help="User question (default: generic summary).",
    )
    args = parser.parse_args()
    inspect(args.rsid.strip().lower(), args.question)


if __name__ == "__main__":
    main()
