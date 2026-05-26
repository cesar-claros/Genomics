"""
Inspection tool for the lookup stage.

Shows the input rsID, the two external lookups, the parsed records, the
disease/biomarker partition that the downstream stages depend on, the
roll-up decision tree (which precedence rule fired and why), and the
summary-trait selection (which tier won).

Pedagogical, not for production use. Makes live HTTP calls just like the
normal lookup, but every intermediate step is printed.

Usage:
  python -m lookup.inspect rs429358
"""

from __future__ import annotations

# Load code/.env BEFORE other imports so NCBI_EMAIL etc. are set.
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import argparse  # noqa: E402
import sys  # noqa: E402

from .clinvar import lookup_clinvar  # noqa: E402
from .combine import _pick_summary_trait  # noqa: E402
from .gwas import lookup_gwas  # noqa: E402
from .heuristics import is_disease_uri, is_measurement_trait  # noqa: E402
from .models import EffectDirection, VariantVerdict  # noqa: E402


def _rule(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def inspect(rsid: str) -> None:
    rsid = rsid.strip().lower()

    _rule(f"INPUT")
    print(f"  rsID: {rsid}")
    print()

    # --- ClinVar ---
    _rule("CLINVAR LOOKUP")
    print("  Path through NCBI E-utilities:")
    print(f'    1. esearch  db=clinvar  term="{rsid}[Variant Name] OR {rsid}"')
    print( "       -> ClinVar VariationID(s)")
    print( "    2. esummary db=clinvar  id=<VariationID>")
    print( "       -> rich record (significance, review status, traits)")
    print()
    print( "  Note: Bio.Entrez.read(validate=False) is required because NCBI")
    print( "  evolves the XML schema faster than Biopython's bundled DTDs.")
    print()

    cv = lookup_clinvar(rsid)
    print("  Parsed ClinVarRecord:")
    for field, value in cv.model_dump().items():
        print(f"    {field:25s} : {value!r}")
    print()

    # --- GWAS ---
    _rule("GWAS CATALOG LOOKUP")
    print("  Path through EBI REST:")
    print(f"    1. GET /singleNucleotidePolymorphisms/{rsid}/associations")
    print( "         ?projection=associationBySnp")
    print( "       -> HAL response with _embedded.associations[]")
    print( "    2. For each association, follow _links.study.href to fetch the")
    print( "       PMID. Cached per study URL within a single call so multiple")
    print( "       associations sharing a study amortize to one fetch.")
    print()

    gwas = lookup_gwas(rsid)
    pmids = sorted({a.pubmed_id for a in gwas if a.pubmed_id})
    print(f"  Associations returned: {len(gwas)}")
    print(f"  Unique PMIDs: {len(pmids)}")
    print()

    if gwas:
        print("  Top 5 by p-value:")
        sorted_gwas = sorted(
            gwas,
            key=lambda g: g.p_value if g.p_value is not None else float("inf"),
        )
        for a in sorted_gwas[:5]:
            print(
                f"    - trait={a.trait!r:<45s} "
                f"p={a.p_value!s:<12s} "
                f"effect={a.effect.value:<15s} "
                f"pmid={a.pubmed_id}"
            )
    print()

    # --- Disease / biomarker partition ---
    _rule("DISEASE / BIOMARKER PARTITION")
    print("  Heuristic: is_disease_uri(uri) checks for 'MONDO_' in the trait URI.")
    print("  Biomarker check: is_measurement_trait(trait) checks for any of")
    print("    measurement, amount, level, concentration, percentage in the name.")
    print()

    diseases = [a for a in gwas if is_disease_uri(a.mapped_trait_uri)]
    others = [a for a in gwas if not is_disease_uri(a.mapped_trait_uri)]
    print(f"  Disease (MONDO) hits: {len(diseases)}")
    for a in sorted(diseases, key=lambda g: g.p_value or float("inf")):
        print(f"    - {a.trait!r}  p={a.p_value!s}  uri={a.mapped_trait_uri}")
    print(f"  Non-disease hits: {len(others)} (top 5 by p-value shown)")
    for a in sorted(others, key=lambda g: g.p_value or float("inf"))[:5]:
        mflag = "[measurement]" if is_measurement_trait(a.trait) else "[other]"
        print(f"    {mflag:<14s} {a.trait!r}  p={a.p_value!s}")
    print()

    # --- Roll-up walkthrough ---
    _rule("ROLL-UP DECISION TREE (_roll_up_effect)")
    print("  Precedence:")
    print("    Rule 1: ClinVar with >=2 stars and directional call wins.")
    print("    Rule 2: GWAS with mixed risk+protective directions -> UNCERTAIN.")
    print("    Rule 3: GWAS with single direction (risk OR protective) wins.")
    print("    Rule 4: Low-confidence ClinVar (any non-UNKNOWN) wins.")
    print("    Rule 5: UNKNOWN.")
    print()

    gwas_effects = {g.effect for g in gwas}
    cv_dir_directional = cv.effect in (EffectDirection.RISK, EffectDirection.PROTECTIVE)
    cv_high_conf = (cv.review_stars or 0) >= 2

    if cv_dir_directional and cv_high_conf:
        chosen = cv.effect
        why = (
            f"Rule 1 FIRED. ClinVar effect={cv.effect.value}, "
            f"review_stars={cv.review_stars} (>=2)."
        )
    elif (
        EffectDirection.RISK in gwas_effects
        and EffectDirection.PROTECTIVE in gwas_effects
    ):
        chosen = EffectDirection.UNCERTAIN
        why = "Rule 2 FIRED. GWAS has both RISK and PROTECTIVE hits."
    elif EffectDirection.RISK in gwas_effects:
        chosen = EffectDirection.RISK
        why = "Rule 3 FIRED. GWAS has RISK hits and no PROTECTIVE hits."
    elif EffectDirection.PROTECTIVE in gwas_effects:
        chosen = EffectDirection.PROTECTIVE
        why = "Rule 3 FIRED. GWAS has PROTECTIVE hits and no RISK hits."
    elif cv.effect != EffectDirection.UNKNOWN:
        chosen = cv.effect
        why = f"Rule 4 FIRED. Falling back to low-confidence ClinVar ({cv.effect.value})."
    else:
        chosen = EffectDirection.UNKNOWN
        why = "Rule 5 FIRED. No source provided a directional signal."

    print(
        f"  ClinVar: effect={cv.effect.value!r}, review_stars={cv.review_stars!r}, "
        f"high_confidence={cv_high_conf}"
    )
    print(f"  GWAS effects present: {sorted(e.value for e in gwas_effects)}")
    print()
    print(f"  Decision: {why}")
    print(f"  overall_effect = {chosen.value}")
    print()

    # --- Summary-trait selection ---
    _rule("SUMMARY-TRAIT SELECTION (_pick_summary_trait)")
    print("  Preference order:")
    print("    1. First ClinVar condition (curated, disease-specific).")
    print("    2. GWAS hit with the lowest p-value among direction-matching,")
    print("       MONDO-URI (disease) hits.")
    print("    3. GWAS hit with the lowest p-value among direction-matching,")
    print("       non-measurement-token traits.")
    print("    4. Any GWAS hit with the lowest p-value (direction-matching if any).")
    print("    5. Fallback: 'reported traits'.")
    print()

    verdict = VariantVerdict(
        query=rsid,
        rsid=rsid,
        clinvar=cv,
        gwas=gwas,
        overall_effect=chosen,
    )
    if cv.conditions:
        print(f"  Tier 1 (ClinVar conditions): {len(cv.conditions)} present -> uses first.")
    else:
        print("  Tier 1 (ClinVar conditions): empty -> skip.")

    if chosen in (EffectDirection.RISK, EffectDirection.PROTECTIVE):
        direction = chosen
        scorable = [g for g in gwas if g.trait and g.p_value is not None]
        directional = [g for g in scorable if g.effect == direction]
        if directional:
            scorable = directional
        tier_disease = [g for g in scorable if is_disease_uri(g.mapped_trait_uri)]
        tier_not_meas = [g for g in scorable if not is_measurement_trait(g.trait)]
        print(
            f"  After direction filter (matching {direction.value}): "
            f"{len(scorable)} of {len(gwas)} hits."
        )
        print(f"  Tier 2 (MONDO disease URI): {len(tier_disease)} candidates.")
        if tier_disease:
            winner = min(tier_disease, key=lambda g: g.p_value)
            print(f"    Tier 2 WINNER: {winner.trait!r} (p={winner.p_value!s})")
        print(f"  Tier 3 (non-measurement traits): {len(tier_not_meas)} candidates.")

    summary = _pick_summary_trait(verdict)
    print(f"  Chosen trait: {summary!r}")
    print()

    # --- Final ---
    _rule("FINAL VERDICT (lookup_variant output shape)")
    print(f"  overall_effect : {chosen.value}")
    print(f"  summary text   : "
          f"rs{rsid[2:]} is associated with ... related to {summary}")
    print(f"  errors         : {verdict.errors}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lookup.inspect",
        description="Walk through the lookup stage step-by-step for one rsID.",
    )
    parser.add_argument("rsid", help="A dbSNP rsID, e.g. rs429358.")
    args = parser.parse_args()
    if not args.rsid.lower().startswith("rs"):
        print(f"warning: {args.rsid!r} does not look like an rsID.", file=sys.stderr)
    inspect(args.rsid)


if __name__ == "__main__":
    main()
