"""
Inspection tool: see how raw text becomes entities.

Runs the same two extractors that feed the verdict layer (scispaCy NER plus
the rsID regex) but prints their raw output side-by-side, so you can see
exactly what each layer contributes. No network calls; this is purely the
NER stage, useful for understanding model behavior on new text without
paying for ClinVar/GWAS lookups.

Usage:
  python -m extract.inspect "We genotyped APOE rs429358 in 200 patients."
  cat abstract.txt | python -m extract.inspect
  python -m extract.inspect --model en_ner_bc5cdr_md "Aspirin treats fever."

Notes:
  - Default model is en_ner_bionlp13cg_md (gene/protein focus, the one
    code/extract/genes.py uses in production).
  - The other model installed in the container is en_ner_bc5cdr_md, which
    has CHEMICAL and DISEASE labels but no gene labels. Pass --model to
    swap and compare what each model "sees" in the same text.
"""

from __future__ import annotations

import argparse
import sys

from .genes import _load_model
from .rsid import _RSID_RE


def _row(start: int, end: int, text: str, label: str) -> str:
    return f"  [{start:>4}-{end:<4}]  {text:<32}  {label}"


def inspect(text: str, model_name: str) -> None:
    rule = "=" * 72

    print(rule)
    print(f"INPUT ({len(text)} chars)")
    print(rule)
    print(text)
    print()

    # --- spaCy NER ---
    print(rule)
    print(f"spaCy entities ({model_name})")
    print(rule)
    print(
        "Biomedical entity labels from the loaded model. Variant identifiers\n"
        "(rsIDs, HGVS) are NOT in this model's label set; see the regex\n"
        "section below for those.\n"
    )

    try:
        nlp = _load_model(model_name)
    except OSError as e:
        print(f"  Failed to load model '{model_name}': {e}")
        print(
            "  Hint: models installed in the container are en_ner_bc5cdr_md\n"
            "  and en_ner_bionlp13cg_md. Pass one of those via --model."
        )
        sys.exit(1)

    doc = nlp(text)

    if doc.ents:
        for ent in doc.ents:
            print(_row(ent.start_char, ent.end_char, ent.text, ent.label_))

        counts: dict[str, int] = {}
        for ent in doc.ents:
            counts[ent.label_] = counts.get(ent.label_, 0) + 1
        print()
        print(
            "  Label counts: "
            + ", ".join(f"{lbl}={n}" for lbl, n in sorted(counts.items()))
        )
    else:
        print("  (no entities found)")
    print()

    # --- rsID regex ---
    print(rule)
    print(r"rsID regex matches (\brs\d+\b)")
    print(rule)
    rsid_hits = list(_RSID_RE.finditer(text))
    if rsid_hits:
        for m in rsid_hits:
            print(_row(m.start(), m.end(), m.group(0), "RSID"))
    else:
        print("  (no rsIDs found)")
    print()

    # --- Combined inline annotation ---
    print(rule)
    print("Annotated text  (entities wrapped as [text]/LABEL)")
    print(rule)
    spans: list[tuple[int, int, str]] = []
    for ent in doc.ents:
        spans.append((ent.start_char, ent.end_char, ent.label_))
    for m in rsid_hits:
        spans.append((m.start(), m.end(), "RSID"))
    spans.sort()

    out: list[str] = []
    cursor = 0
    for start, end, label in spans:
        if start < cursor:
            continue  # skip overlap (e.g. nested entity spans)
        out.append(text[cursor:start])
        out.append(f"[{text[start:end]}]/{label}")
        cursor = end
    out.append(text[cursor:])
    print("".join(out))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extract.inspect",
        description="Show spaCy NER + rsID regex output for arbitrary text.",
    )
    parser.add_argument("text", nargs="?", help="Text to inspect (or pipe via stdin).")
    parser.add_argument(
        "--model",
        default="en_ner_bionlp13cg_md",
        help="scispaCy model (default: en_ner_bionlp13cg_md, gene/protein).",
    )
    args = parser.parse_args()

    text = args.text if args.text else sys.stdin.read()
    inspect(text, args.model)


if __name__ == "__main__":
    main()
