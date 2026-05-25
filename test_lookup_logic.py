"""
Offline tests for the lookup module's pure logic (no network needed).

Run inside the container:
    singularity exec genomic-rag.sif python test_lookup_logic.py

These cover the parsing/normalization/roll-up functions. The actual API calls
(lookup_clinvar / lookup_gwas / lookup_variant) need internet and are tested
separately with a real rsID once your node is online — see the bottom of this
file for that manual check.
"""

from lookup.clinvar import _review_status_to_stars, _significance_to_effect
from lookup.combine import _roll_up_effect, _summarize
from lookup.gwas import _effect_from_or
from lookup.models import ClinVarRecord, EffectDirection, GwasAssociation, VariantVerdict


def test_significance_mapping():
    assert _significance_to_effect("Pathogenic") == EffectDirection.RISK
    assert _significance_to_effect("Likely pathogenic") == EffectDirection.RISK
    assert _significance_to_effect("Benign") == EffectDirection.BENIGN
    assert _significance_to_effect("Likely benign") == EffectDirection.BENIGN
    assert _significance_to_effect("protective") == EffectDirection.PROTECTIVE
    # 'conflicting' must win even though the string contains 'pathogenicity'
    assert _significance_to_effect(
        "Conflicting interpretations of pathogenicity"
    ) == EffectDirection.UNCERTAIN
    assert _significance_to_effect("Uncertain significance") == EffectDirection.UNCERTAIN
    assert _significance_to_effect(None) == EffectDirection.UNKNOWN
    print("[ok] significance -> effect mapping")


def test_star_mapping():
    assert _review_status_to_stars("practice guideline") == 4
    assert _review_status_to_stars("reviewed by expert panel") == 3
    assert _review_status_to_stars("criteria provided, single submitter") == 1
    assert _review_status_to_stars("no assertion criteria provided") == 0
    assert _review_status_to_stars(None) is None
    assert _review_status_to_stars("some unknown status") is None
    print("[ok] review status -> stars")


def test_or_direction():
    assert _effect_from_or(1.5, None) == EffectDirection.RISK
    assert _effect_from_or(0.7, None) == EffectDirection.PROTECTIVE
    assert _effect_from_or(None, 0.3) == EffectDirection.RISK
    assert _effect_from_or(None, -0.3) == EffectDirection.PROTECTIVE
    assert _effect_from_or(None, None) == EffectDirection.UNKNOWN
    print("[ok] odds-ratio / beta -> direction")


def test_rollup_clinvar_high_conf_wins():
    v = VariantVerdict(query="rs429358", rsid="rs429358")
    v.clinvar = ClinVarRecord(
        rsid="rs429358",
        clinical_significance="Pathogenic",
        review_stars=3,
        effect=EffectDirection.RISK,
        conditions=["Alzheimer disease"],
    )
    assert _roll_up_effect(v) == EffectDirection.RISK
    print("[ok] roll-up: high-confidence ClinVar pathogenic wins")


def test_rollup_conflicting_gwas():
    v = VariantVerdict(query="rsX", rsid="rsX")
    v.gwas = [
        GwasAssociation(rsid="rsX", effect=EffectDirection.RISK),
        GwasAssociation(rsid="rsX", effect=EffectDirection.PROTECTIVE),
    ]
    assert _roll_up_effect(v) == EffectDirection.UNCERTAIN
    print("[ok] roll-up: conflicting GWAS directions -> uncertain")


def test_rollup_gwas_fallback():
    # No ClinVar, single-direction GWAS -> that direction.
    v = VariantVerdict(query="rsY", rsid="rsY")
    v.gwas = [GwasAssociation(rsid="rsY", effect=EffectDirection.PROTECTIVE, trait="T2D")]
    assert _roll_up_effect(v) == EffectDirection.PROTECTIVE
    print("[ok] roll-up: GWAS-only fallback")


def test_summary_and_disclaimer():
    v = VariantVerdict(query="rs429358", rsid="rs429358")
    v.clinvar = ClinVarRecord(
        rsid="rs429358", effect=EffectDirection.RISK, conditions=["Alzheimer disease"]
    )
    v.overall_effect = EffectDirection.RISK
    summary = _summarize(v)
    assert "rs429358" in summary and "increased risk" in summary
    # disclaimer must always be present in serialized output
    dumped = v.model_dump()
    assert "disclaimer" in dumped and dumped["disclaimer"]
    print(f"[ok] summary + disclaimer (summary: {summary!r})")


if __name__ == "__main__":
    test_significance_mapping()
    test_star_mapping()
    test_or_direction()
    test_rollup_clinvar_high_conf_wins()
    test_rollup_conflicting_gwas()
    test_rollup_gwas_fallback()
    test_summary_and_disclaimer()
    print("\nAll pure-logic tests passed.")
    print(
        "\nNext, with internet + .env set, run the live check:\n"
        "    python -m lookup.combine rs429358\n"
        "Expect: ClinVar APOE/Alzheimer association, effect=risk_increasing."
    )
