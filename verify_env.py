#!/usr/bin/env python
"""
Environment verification for the genomic variant-disease pipeline.

Run inside the container:
    singularity exec --nv genomic-rag.sif python verify_env.py

Checks are ordered by dependency: foundational pieces (torch + CUDA) first,
so the FIRST failure points at the real cause rather than a downstream symptom.
Each check is isolated in try/except so one missing package doesn't hide the rest.
Exit code is non-zero if any check fails, so it works in CI / job preambles too.
"""

import importlib
import sys
import traceback

# ANSI colors (harmless if the terminal ignores them)
OK = "\033[92m[ OK ]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[96m[INFO]\033[0m"

results = {"ok": 0, "fail": 0, "warn": 0}


def check(label):
    """Decorator: run a check fn, catch anything, tally the result."""
    def wrapper(fn):
        try:
            msg = fn()
            print(f"{OK} {label}: {msg}")
            results["ok"] += 1
        except Exception as e:  # noqa: BLE001 - we want to catch everything here
            print(f"{FAIL} {label}: {type(e).__name__}: {e}")
            results["fail"] += 1
            # Uncomment for full tracebacks while debugging:
            # traceback.print_exc()
        return fn
    return wrapper


def warn(label, msg):
    print(f"{WARN} {label}: {msg}")
    results["warn"] += 1


print("=" * 70)
print("  GENOMIC VARIANT-DISEASE PIPELINE — ENVIRONMENT VERIFICATION")
print("=" * 70)
print(f"{INFO} Python {sys.version.split()[0]} @ {sys.executable}")
print("-" * 70)

# ---------------------------------------------------------------------------
# 1. FOUNDATION: torch + CUDA + the A100 visible inside the container
#    Everything else (vllm, PyG, transformers) sits on this, so it goes first.
# ---------------------------------------------------------------------------
print(f"\n{INFO} --- Foundation: torch + CUDA ---")


@check("numpy (must be <2 for spaCy 3.7 ABI)")
def _():
    import numpy as np
    # spaCy 3.7.x is built against numpy 1.x; numpy 2.x triggers the
    # "numpy.dtype size changed" ABI crash. Flag it here before spaCy does.
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        warn("numpy", f"{np.__version__} is 2.x — spaCy 3.7 will crash; pin numpy<2")
    return f"numpy {np.__version__}"


@check("torch import + version")
def _():
    import torch
    return f"torch {torch.__version__}"


@check("CUDA available to torch")
def _():
    import torch
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False — did you launch with `--nv`, "
        "and is this a GPU node?"
    )
    return f"CUDA {torch.version.cuda}, devices={torch.cuda.device_count()}"


@check("GPU is the expected A100")
def _():
    import torch
    name = torch.cuda.get_device_name(0)
    if "A100" not in name:
        warn("GPU model", f"expected A100, got '{name}' (not fatal)")
    cap = torch.cuda.get_device_capability(0)
    return f"{name}, compute capability {cap[0]}.{cap[1]}"


@check("bf16 supported (A100 should be True)")
def _():
    import torch
    supported = torch.cuda.is_bf16_supported()
    if not supported:
        warn("bf16", "not supported — fall back to fp16 in training/inference")
    return f"bf16_supported={supported}"


@check("actual GPU compute (matmul on device)")
def _():
    import torch
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = (a @ b).sum().item()  # forces real kernel execution + sync
    free, total = torch.cuda.mem_get_info()
    return f"matmul ok; VRAM free={free // 2**30}GB / total={total // 2**30}GB"


# ---------------------------------------------------------------------------
# 2. CORE ML / NLP STACK
# ---------------------------------------------------------------------------
print(f"\n{INFO} --- Core ML / NLP ---")


@check("transformers + token-classification head")
def _():
    import transformers
    from transformers import AutoModelForTokenClassification  # noqa: F401
    return f"transformers {transformers.__version__}"


@check("datasets")
def _():
    import datasets
    return f"datasets {datasets.__version__}"


@check("accelerate")
def _():
    import accelerate
    return f"accelerate {accelerate.__version__}"


@check("peft (LoRA/QLoRA)")
def _():
    import peft
    return f"peft {peft.__version__}"


@check("bitsandbytes + CUDA backend")
def _():
    import bitsandbytes as bnb
    # bitsandbytes silently degrades if its CUDA .so didn't load; probe it.
    has_cuda = getattr(bnb.functional, "CUBLAS_Context", None) is not None
    return f"bitsandbytes {bnb.__version__} (cuda_ext_loaded≈{has_cuda})"


@check("seqeval (NER metrics)")
def _():
    from seqeval.metrics import f1_score  # noqa: F401
    return "seqeval import ok"


# ---------------------------------------------------------------------------
# 3. REASONING LAYER: vllm (local) + hosted SDKs
#    vllm import also re-validates the torch build it pinned.
# ---------------------------------------------------------------------------
print(f"\n{INFO} --- Reasoning layer ---")


@check("reasoning backend: hosted API clients present")
def _():
    # Local vllm serving was deferred (see pyproject note). The reasoning layer
    # runs through hosted APIs for now, so we just confirm the clients import.
    # When vllm is reintroduced later, add an import check for it here.
    import anthropic  # noqa: F401
    import openai  # noqa: F401
    return "anthropic + openai clients import ok (vllm deferred)"


@check("flash-attn (optional — absence is fine)")
def _():
    try:
        import flash_attn
        return f"flash_attn {flash_attn.__version__}"
    except ImportError:
        return "not installed (optional speedup; fine to omit)"


# ---------------------------------------------------------------------------
# 4. BIOMEDICAL NER: spaCy + scispaCy + the model wheel
# ---------------------------------------------------------------------------
print(f"\n{INFO} --- Biomedical NER ---")


@check("spaCy")
def _():
    import spacy
    return f"spaCy {spacy.__version__}"


@check("scispaCy")
def _():
    import scispacy  # noqa: F401
    return "scispacy import ok"


@check("scispaCy NER model loads + runs")
def _():
    import spacy
    # This is the real test: the model wheel installed separately must load
    # AND be able to tag entities. Loading alone isn't enough.
    nlp = spacy.load("en_ner_bc5cdr_md")
    doc = nlp("The patient was treated with metformin for diabetes.")
    ents = [(e.text, e.label_) for e in doc.ents]
    assert ents, "model loaded but found no entities — wheel may be mismatched"
    return f"entities={ents}"


# ---------------------------------------------------------------------------
# 5. GENOMICS: the packages that compiled against htslib system libs.
#    Import success here confirms the libhts/zlib/bz2/lzma build worked.
# ---------------------------------------------------------------------------
print(f"\n{INFO} --- Genomics (htslib-linked) ---")


@check("pysam (links htslib)")
def _():
    import pysam
    # Don't probe libchtslib.__version__ — it isn't present in all builds.
    # Confirm the import worked and a real htslib-backed class is available,
    # which is the actual signal that the htslib system-lib build succeeded.
    assert hasattr(pysam, "AlignmentFile"), "pysam imported but htslib API missing"
    return f"pysam {pysam.__version__} (htslib API present)"


@check("cyvcf2 (links htslib)")
def _():
    import cyvcf2
    return f"cyvcf2 {cyvcf2.__version__}"


@check("biopython (Bio.Entrez for NCBI/ClinVar)")
def _():
    import Bio
    from Bio import Entrez  # noqa: F401
    return f"biopython {Bio.__version__}"


@check("myvariant (variant annotation client)")
def _():
    import myvariant  # noqa: F401
    return "myvariant import ok"


# ---------------------------------------------------------------------------
# 6. KNOWLEDGE GRAPH: driver import + torch-geometric ABI alignment.
#    PyG is the package most likely to break against the wrong torch build,
#    so importing torch_scatter/torch_sparse is the real ABI test.
# ---------------------------------------------------------------------------
print(f"\n{INFO} --- Knowledge graph ---")


@check("neo4j driver")
def _():
    import neo4j
    return f"neo4j driver {neo4j.__version__}"


@check("torch-geometric + compiled extensions (ABI test)")
def _():
    import torch_geometric
    # These two are the compiled companions; if torch was swapped under PyG,
    # THESE are what throw the cryptic ABI/symbol errors.
    import torch_scatter  # noqa: F401
    import torch_sparse  # noqa: F401
    return f"torch_geometric {torch_geometric.__version__} (+ scatter/sparse ok)"


# ---------------------------------------------------------------------------
# 7. DATA / SERVING / ORCHESTRATION
# ---------------------------------------------------------------------------
print(f"\n{INFO} --- Data / serving / orchestration ---")


@check("pandas + pyarrow + polars")
def _():
    import pandas as pd
    import pyarrow
    import polars as pl
    return f"pandas {pd.__version__}, pyarrow {pyarrow.__version__}, polars {pl.__version__}"


@check("langchain + provider integrations")
def _():
    import langchain
    import langchain_anthropic  # noqa: F401
    import langchain_openai  # noqa: F401
    return f"langchain {langchain.__version__} (+ anthropic/openai integrations)"


@check("fastapi + uvicorn")
def _():
    import fastapi
    import uvicorn
    return f"fastapi {fastapi.__version__}, uvicorn {uvicorn.__version__}"


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"  RESULTS: {results['ok']} ok, {results['fail']} failed, {results['warn']} warnings")
print("=" * 70)

if results["fail"]:
    print(f"{FAIL} Environment has problems — see failures above (top-most is root cause).")
    sys.exit(1)
else:
    print(f"{OK} Environment looks good. Safe to start writing pipeline code.")
    sys.exit(0)
