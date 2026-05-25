"""
ClinVar lookup via NCBI E-utilities.

Flow for an rsID:
  1. esearch in the 'clinvar' db for the rsID  -> ClinVar VariationID(s)
  2. esummary for that VariationID             -> clinical significance, review
                                                  status, associated conditions

We use biopython's Bio.Entrez wrapper (already a project dependency). NCBI asks
that every request include an email; an API key raises the rate limit from 3 to
10 requests/sec. Both come from the environment (.env): NCBI_EMAIL, NCBI_API_KEY.

NOTE: requires outbound internet on the machine that runs it. Many HPC *compute*
nodes have no internet — run from a login node, or use the bulk-download path,
if esearch times out.
"""

from __future__ import annotations

import os

from Bio import Entrez

from .models import ClinVarRecord, EffectDirection

# Configure Entrez from environment. NCBI requires an email; key is optional.
Entrez.email = os.environ.get("NCBI_EMAIL", "")
_api_key = os.environ.get("NCBI_API_KEY")
if _api_key:
    Entrez.api_key = _api_key


# Map ClinVar's free-text significance to our normalized effect direction.
# ClinVar significance strings are messy and sometimes compound
# ("Pathogenic/Likely pathogenic"), so we match on lowercase substrings.
def _significance_to_effect(significance: str | None) -> EffectDirection:
    if not significance:
        return EffectDirection.UNKNOWN
    s = significance.lower()
    # Order matters: check 'conflicting'/'uncertain' before pathogenic/benign,
    # since a conflicting record may contain both words.
    if "conflicting" in s or "uncertain" in s:
        return EffectDirection.UNCERTAIN
    if "pathogenic" in s:        # covers 'pathogenic' and 'likely pathogenic'
        return EffectDirection.RISK
    if "protective" in s:
        return EffectDirection.PROTECTIVE
    if "benign" in s:            # covers 'benign' and 'likely benign'
        return EffectDirection.BENIGN
    return EffectDirection.UNCERTAIN


# ClinVar review status -> star rating (mirrors ClinVar's own 0-4 star scheme).
_REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
}


def _review_status_to_stars(review_status: str | None) -> int | None:
    if not review_status:
        return None
    return _REVIEW_STARS.get(review_status.strip().lower())


def lookup_clinvar(rsid: str, timeout: int = 20) -> ClinVarRecord:
    """
    Look up a single rsID in ClinVar. Returns a ClinVarRecord; on miss or error
    returns a record with effect=UNKNOWN (caller can inspect for emptiness).

    Raises nothing for "not found" — only network/parse issues propagate, and
    the caller (combine.py) wraps this so one source failing doesn't sink the
    whole verdict.
    """
    rsid = rsid.strip()
    # ClinVar indexes dbSNP IDs without the 'rs' prefix under the snp filter;
    # searching the bare rsID term works well in practice.
    search_term = f"{rsid}[Variant Name] OR {rsid}"

    handle = Entrez.esearch(db="clinvar", term=search_term, retmax=1)
    search = Entrez.read(handle)
    handle.close()

    id_list = search.get("IdList", [])
    if not id_list:
        return ClinVarRecord(rsid=rsid, effect=EffectDirection.UNKNOWN)

    variation_id = id_list[0]

    # esummary returns the rich record (significance, review status, conditions).
    handle = Entrez.esummary(db="clinvar", id=variation_id)
    summary = Entrez.read(handle)
    handle.close()

    # The esummary structure for clinvar nests the useful bits under
    # DocumentSummarySet -> DocumentSummary[0]. Guard every access because the
    # shape varies by record and NCBI occasionally changes field names.
    try:
        doc = summary["DocumentSummarySet"]["DocumentSummary"][0]
    except (KeyError, IndexError, TypeError):
        return ClinVarRecord(
            rsid=rsid,
            clinvar_variation_id=str(variation_id),
            effect=EffectDirection.UNKNOWN,
        )

    # germline_classification holds the modern significance + review status.
    # Fall back to older field names for robustness across record vintages.
    significance = None
    review_status = None
    germ = doc.get("germline_classification")
    if isinstance(germ, dict):
        significance = germ.get("description")
        review_status = germ.get("review_status")
    if significance is None:
        significance = doc.get("clinical_significance", {}).get("description") \
            if isinstance(doc.get("clinical_significance"), dict) else None

    # Conditions: ClinVar lists trait names under germline_classification ->
    # trait_set, or the older 'trait_set' at the top level.
    conditions: list[str] = []
    trait_set = None
    if isinstance(germ, dict):
        trait_set = germ.get("trait_set")
    if trait_set is None:
        trait_set = doc.get("trait_set")
    if isinstance(trait_set, list):
        for trait in trait_set:
            if isinstance(trait, dict):
                name = trait.get("trait_name") or trait.get("name")
                if name:
                    conditions.append(name)

    return ClinVarRecord(
        rsid=rsid,
        clinvar_variation_id=str(variation_id),
        clinical_significance=significance,
        review_status=review_status,
        review_stars=_review_status_to_stars(review_status),
        conditions=conditions,
        effect=_significance_to_effect(significance),
    )
