"""
GWAS Catalog lookup via the EBI REST API.

The GWAS Catalog exposes a REST endpoint keyed by rsID:
    GET /gwas/rest/api/singleNucleotidePolymorphisms/{rsid}/associations

Each association carries a p-value, optionally an effect size (OR or beta), the
risk allele, the reported trait, and the source study (PMID), which we keep for
citation. The Catalog is association data (population-level), so effect direction
here means "risk-increasing vs protective" inferred from OR / beta when present.

NOTE: requires outbound internet. No key needed. We keep timeouts tight and fail
soft — a GWAS miss should never sink a verdict that ClinVar could still answer.
"""

from __future__ import annotations

import requests

from .models import EffectDirection, GwasAssociation

_BASE = "https://www.ebi.ac.uk/gwas/rest/api"


def _effect_from_or(odds_ratio: float | None, beta: float | None) -> EffectDirection:
    """
    Infer direction from effect size:
      - OR > 1  => risk-increasing;  OR < 1 => protective
      - beta > 0 => risk-increasing; beta < 0 => protective (trait-dependent,
        so this is a heuristic the reasoning layer can refine)
    """
    if odds_ratio is not None:
        if odds_ratio > 1.0:
            return EffectDirection.RISK
        if odds_ratio < 1.0:
            return EffectDirection.PROTECTIVE
    if beta is not None:
        if beta > 0:
            return EffectDirection.RISK
        if beta < 0:
            return EffectDirection.PROTECTIVE
    return EffectDirection.UNKNOWN


def lookup_gwas(rsid: str, timeout: int = 20, max_results: int = 20) -> list[GwasAssociation]:
    """
    Return GWAS Catalog associations for an rsID. Empty list on miss.
    Network/HTTP errors propagate to the caller, which wraps them.
    """
    rsid = rsid.strip()
    url = f"{_BASE}/singleNucleotidePolymorphisms/{rsid}/associations"
    # projection=associationBySnp gives a flatter, easier-to-parse shape.
    resp = requests.get(url, params={"projection": "associationBySnp"}, timeout=timeout)

    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    data = resp.json()
    # Associations live under _embedded.associations (HAL/HATEOAS format).
    assocs = data.get("_embedded", {}).get("associations", [])

    results: list[GwasAssociation] = []
    for a in assocs[:max_results]:
        p_value = a.get("pvalue")
        odds_ratio = a.get("orPerCopyNum")  # OR per copy, when reported
        beta = a.get("betaNum")

        # Risk allele + trait can be nested under loci/strongestRiskAlleles.
        risk_allele = None
        loci = a.get("loci") or []
        if loci:
            alleles = loci[0].get("strongestRiskAlleles") or []
            if alleles:
                risk_allele = alleles[0].get("riskAlleleName")

        # Reported trait(s)
        trait = None
        mapped_uri = None
        efo_traits = a.get("efoTraits") or []
        if efo_traits:
            trait = efo_traits[0].get("trait")
            mapped_uri = efo_traits[0].get("uri")

        # Source study PMID for citation.
        pmid = None
        study = a.get("study") or {}
        pub = study.get("publicationInfo") or {}
        pmid = pub.get("pubmedId")

        results.append(
            GwasAssociation(
                rsid=rsid,
                trait=trait,
                mapped_trait_uri=mapped_uri,
                p_value=float(p_value) if p_value is not None else None,
                odds_ratio=float(odds_ratio) if odds_ratio is not None else None,
                risk_allele=risk_allele,
                effect=_effect_from_or(
                    float(odds_ratio) if odds_ratio is not None else None,
                    float(beta) if beta is not None else None,
                ),
                pubmed_id=str(pmid) if pmid is not None else None,
            )
        )

    return results
