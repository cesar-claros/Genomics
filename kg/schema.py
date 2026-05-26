"""
Knowledge-graph schema: node labels, relationship types, uniqueness constraints.

Variant-centric design. Each VariantVerdict from the lookup layer maps to:

  (Variant {rsid})
      -[:IN_GENE]-> (Gene {symbol})
      -[:HAS_CLINVAR_CONDITION {review_stars, significance, effect}]
            -> (Disease {name})
      -[:HAS_GWAS_ASSOCIATION {p_value, odds_ratio, effect, risk_allele, pmid}]
            -> (Disease|Trait {name, uri})
      -[:CITED_IN]-> (Study {pmid})

Disease vs Trait split is driven by the `is_disease_uri` heuristic from
lookup/heuristics.py: GWAS hits with a MONDO URI become Disease nodes;
everything else (EFO measurement, OBA, etc.) becomes Trait. ClinVar
conditions are always Disease since ClinVar's mandate is clinical
disease curation.

Notes on duplication: ClinVar's free-text disease names ("Breast-ovarian
cancer, familial, susceptibility to, 1") and GWAS's ontology names
("Alzheimer disease") will produce separate Disease nodes for what is
semantically the same condition. Cross-source disease linking is a real
entity-linking task; deferred to a later iteration.
"""

from __future__ import annotations

from enum import Enum


class NodeLabel(str, Enum):
    VARIANT = "Variant"
    GENE = "Gene"
    DISEASE = "Disease"
    TRAIT = "Trait"
    STUDY = "Study"


class RelType(str, Enum):
    IN_GENE = "IN_GENE"
    HAS_CLINVAR_CONDITION = "HAS_CLINVAR_CONDITION"
    HAS_GWAS_ASSOCIATION = "HAS_GWAS_ASSOCIATION"
    CITED_IN = "CITED_IN"


# Cypher to create uniqueness constraints. Idempotent (IF NOT EXISTS),
# safe to run on every connection. Required for MERGE to behave correctly:
# without these, MERGE on a key property does a full label scan and may
# create duplicates under concurrent writes.
CONSTRAINTS_CYPHER = [
    "CREATE CONSTRAINT variant_rsid IF NOT EXISTS "
    "FOR (v:Variant) REQUIRE v.rsid IS UNIQUE",
    "CREATE CONSTRAINT gene_symbol IF NOT EXISTS "
    "FOR (g:Gene) REQUIRE g.symbol IS UNIQUE",
    "CREATE CONSTRAINT disease_name IF NOT EXISTS "
    "FOR (d:Disease) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT trait_name IF NOT EXISTS "
    "FOR (t:Trait) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT study_pmid IF NOT EXISTS "
    "FOR (s:Study) REQUIRE s.pmid IS UNIQUE",
]
