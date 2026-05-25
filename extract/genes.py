"""
Gene mention extraction via scispaCy.

Default model is en_ner_bionlp13cg_md, whose label set includes
GENE_OR_GENE_PRODUCT (plus AMINO_ACID, CELL_LINE, ORGAN, etc., which we
ignore here). The model is installed in container/Dockerfile alongside the
existing en_ner_bc5cdr_md (chemicals + diseases).

Notes for callers:
  - The spaCy model is loaded lazily on first call and cached. Cold load is
    ~1-3 s; subsequent calls are millisecond-scale plus per-document NER cost.
  - Override the model at runtime by setting the GENE_NER_MODEL env var, or
    pass `model_name=` explicitly. Useful for swapping in en_ner_jnlpba_md
    or a fine-tuned model later.
  - No canonicalization yet: `symbol` is just the uppercased surface form.
    HGNC alias resolution lands in the entity-linking iteration.
"""

from __future__ import annotations

import os
from functools import lru_cache

from .models import GeneMention

_DEFAULT_MODEL = "en_ner_bionlp13cg_md"

# bionlp13cg labels we treat as a gene mention. AMINO_ACID and similar
# molecule-level labels are excluded; they overlap with diseases / chemistry
# and confuse the downstream gene -> variant association heuristic.
_GENE_LABELS = frozenset({"GENE_OR_GENE_PRODUCT"})


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    # Lazy import so `from extract import ...` doesn't pay spaCy's import cost
    # when the caller only wants the rsID regex.
    import spacy

    return spacy.load(model_name)


def extract_genes(text: str, model_name: str | None = None) -> list[GeneMention]:
    """Run scispaCy NER on `text`; return only gene-label mentions, in source order."""
    name = model_name or os.environ.get("GENE_NER_MODEL") or _DEFAULT_MODEL
    nlp = _load_model(name)
    doc = nlp(text)

    out: list[GeneMention] = []
    for ent in doc.ents:
        if ent.label_ not in _GENE_LABELS:
            continue
        surface = ent.text
        out.append(
            GeneMention(
                text=surface,
                symbol=surface.upper(),
                start=ent.start_char,
                end=ent.end_char,
                label=ent.label_,
            )
        )
    return out
