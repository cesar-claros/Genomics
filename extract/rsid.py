"""
rsID extraction via regex.

dbSNP rsIDs are stable, simply-shaped tokens ('rs' followed by digits), so
pattern matching is more reliable here than any NER model. Word boundaries
prevent matches inside other identifiers (e.g. 'somers12345' will not match).

Case-insensitive: occasional uppercase 'RS429358' shows up in clinical text.
Mentions are normalized to lowercase in the rsid field; the original surface
form is kept in `text` so downstream can quote it.
"""

from __future__ import annotations

import re

from .models import VariantMention

_RSID_RE = re.compile(r"\brs\d+\b", re.IGNORECASE)


def extract_rsids(text: str) -> list[VariantMention]:
    """Find every rsID-shaped token in `text`. Mentions are returned in source order."""
    out: list[VariantMention] = []
    for m in _RSID_RE.finditer(text):
        surface = m.group(0)
        out.append(
            VariantMention(
                text=surface,
                rsid=surface.lower(),
                start=m.start(),
                end=m.end(),
            )
        )
    return out
