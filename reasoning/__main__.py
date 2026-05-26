"""CLI entrypoint: `python -m reasoning <text-or-rsid> [--question Q]`.

If the input matches an rsID exactly, looks it up directly and synthesizes.
Otherwise treats the input as free text, extracts variant mentions with the
NER stage, looks each up, and synthesizes an answer per variant.

Examples:
  python -m reasoning rs429358
  python -m reasoning --question "Is rs429358 protective in any context?" rs429358
  python -m reasoning "We found APOE rs429358 in patients with early-onset AD."

Environment:
  REASONING_MODEL=<hf_model_id>   (default: microsoft/Phi-3.5-mini-instruct)
  REASONING_DTYPE=fp16|bf16|fp32  (default: fp16)
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from extract.combine import extract_and_lookup
from lookup.combine import lookup_variant

from .synthesize import synthesize

_RSID_ONLY_RE = re.compile(r"^\s*rs\d+\s*$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reasoning",
        description="LLM-synthesized answer grounded in variant evidence.",
    )
    parser.add_argument("input", nargs="?", help="An rsID or free-text passage (or pipe via stdin).")
    parser.add_argument(
        "--question",
        default=None,
        help="User question (default: generic evidence summary).",
    )
    args = parser.parse_args()

    text = args.input if args.input else sys.stdin.read().strip()

    if _RSID_ONLY_RE.match(text):
        verdict = lookup_variant(text.strip().lower())
        answer = synthesize(verdict, question=args.question)
        print(json.dumps(answer.model_dump(), indent=2, default=str))
        return

    pipeline = extract_and_lookup(text)
    payload = {
        "input": text,
        "mentions": pipeline.mentions.model_dump(),
        "answers": [],
    }
    for v in pipeline.verdicts:
        ans = synthesize(v, question=args.question, mention_context=text)
        payload["answers"].append(ans.model_dump())
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
