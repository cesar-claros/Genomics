"""
Side-by-side NER comparison: scispaCy (CNN + word vectors) vs. a finetuned
BERT (transformer encoder).

Runs both models on the same text and prints:
  - Each model's raw entities (surface form, char span, label, score)
  - An agreement summary: entities both find, only-spaCy, only-BERT
  - Timing: load-plus-first-inference and warm inference, per model

Defaults to scispaCy en_ner_bionlp13cg_md and pruas/BENT-PubMedBERT-NER-Gene.
Both are configurable via CLI flags so you can sweep across HuggingFace
models without rewriting the tool.

Usage:
  python -m extract.compare "We genotyped APOE rs429358 in 200 patients."
  python -m extract.compare --bert-model alvaroalon2/biobert_genetic_ner "..."
  cat abstract.txt | python -m extract.compare

Notes:
  - First call to a BERT model downloads it from HuggingFace (~440 MB for
    PubMedBERT-base) into ~/.cache/huggingface. Subsequent calls are fast.
  - The BERT model runs on GPU if torch.cuda.is_available(), else CPU.
  - Agreement is computed by exact char-span match. Span disagreements
    (e.g. one model includes trailing punctuation, the other doesn't) show
    up as separate "only-X" rows so you can spot them.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

from .genes import _load_model

_DEFAULT_BERT = "pruas/BENT-PubMedBERT-NER-Gene"
_DEFAULT_SPACY = "en_ner_bionlp13cg_md"


@dataclass(frozen=True)
class Entity:
    """Normalized entity record. `score` is None for scispaCy (no confidence)."""

    start: int
    end: int
    text: str
    label: str
    score: float | None = None


@lru_cache(maxsize=4)
def _load_bert(model_name: str):
    """Lazy-load a HuggingFace token-classification pipeline. Cached per name.
    aggregation_strategy='simple' merges subword B-/I- tags into entity spans
    with character offsets."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="ner",
        model=model_name,
        aggregation_strategy="simple",
        device=device,
    )


def _spacy_entities(text: str, model_name: str) -> tuple[list[Entity], float]:
    nlp = _load_model(model_name)
    t0 = time.perf_counter()
    doc = nlp(text)
    dt = time.perf_counter() - t0
    ents = [Entity(e.start_char, e.end_char, e.text, e.label_, None) for e in doc.ents]
    return ents, dt


def _bert_entities(text: str, model_name: str) -> tuple[list[Entity], float]:
    pipe = _load_bert(model_name)

    # Some single-type HF models (e.g. BENT) train with raw BIO labels
    # (O / B / I, no entity-type suffix). aggregation_strategy='simple' has
    # nothing after the '-' to strip, so entity_group comes out as 'B'.
    # When we detect that, infer the type from the trailing segment of the
    # model name (BENT convention: NER-<Type>).
    id2label = list(getattr(pipe.model.config, "id2label", {}).values())
    bio_only = bool(id2label) and set(id2label).issubset({"O", "B", "I"})
    inferred_type = model_name.rsplit("-", 1)[-1].upper() if bio_only else None

    t0 = time.perf_counter()
    raw = pipe(text)
    dt = time.perf_counter() - t0

    ents: list[Entity] = []
    for r in raw:
        group = str(r["entity_group"])
        label = inferred_type if (bio_only and group in {"B", "I"}) else group
        ents.append(
            Entity(
                start=int(r["start"]),
                end=int(r["end"]),
                # Slice the original text rather than trusting BERT's `word`
                # (uncased models lowercase it; subword merges may add spaces).
                text=text[int(r["start"]):int(r["end"])],
                label=label,
                score=float(r["score"]),
            )
        )
    return ents, dt


def _row(e: Entity) -> str:
    score_str = f"{e.score:.3f}" if e.score is not None else "  -  "
    return f"  [{e.start:>4}-{e.end:<4}]  {e.text:<28}  {e.label:<24}  {score_str}"


def _print_entities(title: str, entities: list[Entity]) -> None:
    rule = "=" * 80
    print(rule)
    print(title)
    print(rule)
    if not entities:
        print("  (no entities found)")
    else:
        for e in entities:
            print(_row(e))
    print()


def compare(text: str, spacy_model: str, bert_model: str) -> None:
    rule = "=" * 80

    print(rule)
    print(f"INPUT ({len(text)} chars)")
    print(rule)
    print(text)
    print()

    print(f"Loading scispaCy: {spacy_model} ...")
    t0 = time.perf_counter()
    spacy_ents, spacy_dt = _spacy_entities(text, spacy_model)
    spacy_total = time.perf_counter() - t0
    print(
        f"  done. load + first inference: {spacy_total:.2f}s "
        f"(pure inference: {spacy_dt * 1000:.1f} ms)\n"
    )

    print(f"Loading BERT:     {bert_model} ...")
    t0 = time.perf_counter()
    bert_ents, bert_dt = _bert_entities(text, bert_model)
    bert_total = time.perf_counter() - t0
    print(
        f"  done. load + first inference: {bert_total:.2f}s "
        f"(pure inference: {bert_dt * 1000:.1f} ms)\n"
    )

    _print_entities(f"scispaCy entities ({spacy_model})", spacy_ents)
    _print_entities(f"BERT entities     ({bert_model})", bert_ents)

    # Agreement analysis by exact char-span match.
    spacy_by_span = {(e.start, e.end): e for e in spacy_ents}
    bert_by_span = {(e.start, e.end): e for e in bert_ents}
    both = sorted(spacy_by_span.keys() & bert_by_span.keys())
    only_spacy = sorted(spacy_by_span.keys() - bert_by_span.keys())
    only_bert = sorted(bert_by_span.keys() - spacy_by_span.keys())

    print(rule)
    print("AGREEMENT SUMMARY  (exact-span match)")
    print(rule)
    print(f"  Both find:     {len(both):>3}")
    print(f"  Only scispaCy: {len(only_spacy):>3}")
    print(f"  Only BERT:     {len(only_bert):>3}")
    print()

    if both:
        print("  Both find  (span, text, spaCy label / BERT label, BERT score):")
        for span in both:
            s = spacy_by_span[span]
            b = bert_by_span[span]
            print(
                f"    {span}  {s.text!r:<30}  {s.label}  /  {b.label} "
                f"({b.score:.3f})"
            )
        print()

    if only_spacy:
        print("  Only scispaCy:")
        for span in only_spacy:
            e = spacy_by_span[span]
            print(f"    {span}  {e.text!r:<30}  {e.label}")
        print()

    if only_bert:
        print("  Only BERT:")
        for span in only_bert:
            e = bert_by_span[span]
            print(f"    {span}  {e.text!r:<30}  {e.label}  (score={e.score:.3f})")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extract.compare",
        description="Compare scispaCy NER against a finetuned BERT NER model.",
    )
    parser.add_argument("text", nargs="?", help="Text to compare (or pipe via stdin).")
    parser.add_argument(
        "--spacy-model",
        default=_DEFAULT_SPACY,
        help=f"scispaCy model name (default: {_DEFAULT_SPACY}).",
    )
    parser.add_argument(
        "--bert-model",
        default=_DEFAULT_BERT,
        help=f"HuggingFace model name (default: {_DEFAULT_BERT}).",
    )
    args = parser.parse_args()
    text = args.text if args.text else sys.stdin.read()
    compare(text, args.spacy_model, args.bert_model)


if __name__ == "__main__":
    main()
