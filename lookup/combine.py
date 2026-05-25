"""
Combine ClinVar + GWAS lookups into a single VariantVerdict.

Design choices:
  - FAIL SOFT: each source is wrapped in try/except so one source erroring
    (or the node lacking internet for one endpoint) still yields a partial,
    usable verdict with the error recorded in `errors`.
  - The roll-up `overall_effect` is intentionally COARSE. Resolving genuine
    conflicts between sources (ClinVar says benign, a GWAS says weak risk) is
    the reasoning layer's job, not a hard-coded rule here. We just surface a
    sensible default and let the nuanced synthesis happen downstream.
"""

from __future__ import annotations

from .clinvar import lookup_clinvar
from .gwas import lookup_gwas
from .models import EffectDirection, VariantVerdict


def _roll_up_effect(verdict: VariantVerdict) -> EffectDirection:
    """
    Coarse precedence for a single display-level effect:
      1. A high-confidence ClinVar classification (>=2 stars) wins.
      2. Otherwise, a clear GWAS direction (any risk/protective) is used.
      3. Otherwise fall back to whatever ClinVar said (even low-confidence).
      4. Else uncertain/unknown.
    """
    cv = verdict.clinvar
    if cv and cv.effect in (EffectDirection.RISK, EffectDirection.PROTECTIVE):
        if (cv.review_stars or 0) >= 2:
            return cv.effect

    gwas_effects = {g.effect for g in verdict.gwas}
    if EffectDirection.RISK in gwas_effects and EffectDirection.PROTECTIVE in gwas_effects:
        return EffectDirection.UNCERTAIN  # conflicting GWAS directions
    if EffectDirection.RISK in gwas_effects:
        return EffectDirection.RISK
    if EffectDirection.PROTECTIVE in gwas_effects:
        return EffectDirection.PROTECTIVE

    if cv and cv.effect != EffectDirection.UNKNOWN:
        return cv.effect
    return EffectDirection.UNKNOWN


def _pick_summary_trait(verdict: VariantVerdict) -> str:
    """
    Choose the most informative trait for the one-line summary.

    Preference:
      1. First curated ClinVar condition (disease-specific).
      2. GWAS trait with the lowest p-value among associations whose direction
         matches the overall_effect (when that effect is RISK or PROTECTIVE).
         This avoids picking an arbitrary first-in-list trait for pleiotropic
         variants like APOE, where the first hit may be a weak measurement
         while a much stronger disease association is buried further down.
      3. GWAS trait with the lowest p-value overall.
      4. The literal "reported traits" fallback.
    """
    cv = verdict.clinvar
    if cv and cv.conditions:
        return cv.conditions[0]

    if verdict.gwas:
        scorable = [g for g in verdict.gwas if g.trait and g.p_value is not None]
        direction = verdict.overall_effect
        if direction in (EffectDirection.RISK, EffectDirection.PROTECTIVE):
            directional = [g for g in scorable if g.effect == direction]
            if directional:
                scorable = directional
        if scorable:
            return min(scorable, key=lambda g: g.p_value).trait
        for g in verdict.gwas:
            if g.trait:
                return g.trait

    return "reported traits"


def _summarize(verdict: VariantVerdict) -> str:
    """A neutral one-liner. Deliberately avoids clinical/imperative language."""
    rsid = verdict.rsid or verdict.query
    effect = verdict.overall_effect
    cond_str = _pick_summary_trait(verdict)

    phrasing = {
        EffectDirection.RISK: f"{rsid} is associated with increased risk related to {cond_str}",
        EffectDirection.PROTECTIVE: f"{rsid} is associated with reduced risk related to {cond_str}",
        EffectDirection.BENIGN: f"{rsid} appears benign / no significant disease association reported",
        EffectDirection.UNCERTAIN: f"{rsid} has uncertain or conflicting evidence for {cond_str}",
        EffectDirection.UNKNOWN: f"No association for {rsid} was found in the queried sources",
    }
    return phrasing[effect]


def lookup_variant(
    rsid: str,
    use_clinvar: bool = True,
    use_gwas: bool = True,
    gene: str | None = None,
) -> VariantVerdict:
    """
    Look up a variant across ClinVar + GWAS and return a combined verdict.

    `rsid` should be a normalized dbSNP rsID (e.g. 'rs429358'). Entity
    normalization (text mention -> rsID) is a separate, upstream concern.
    """
    verdict = VariantVerdict(query=rsid, rsid=rsid.strip(), gene=gene)

    if use_clinvar:
        verdict.sources_queried.append("clinvar")
        try:
            verdict.clinvar = lookup_clinvar(rsid)
        except Exception as e:  # noqa: BLE001 - fail soft, record and continue
            verdict.errors.append(f"clinvar: {type(e).__name__}: {e}")

    if use_gwas:
        verdict.sources_queried.append("gwas_catalog")
        try:
            verdict.gwas = lookup_gwas(rsid)
        except Exception as e:  # noqa: BLE001 - fail soft
            verdict.errors.append(f"gwas: {type(e).__name__}: {e}")

    verdict.overall_effect = _roll_up_effect(verdict)
    verdict.summary = _summarize(verdict)
    return verdict
