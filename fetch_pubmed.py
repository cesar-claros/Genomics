"""
Resolve DOIs to PubMed abstracts via NCBI E-utilities.

Library and CLI for the two-step lookup (DOI -> PMID via esearch, then
PMID -> abstract text via efetch). Reads NCBI_EMAIL and NCBI_API_KEY
from the environment (or code/.env) for politeness and rate-limit
elevation per NCBI's usage policy.

Library use:

    from fetch_pubmed import fetch_abstract_by_doi
    pmid, text = fetch_abstract_by_doi("10.1503/cmaj.180066")

CLI use:

    # Single DOI to stdout:
    python -m fetch_pubmed 10.1503/cmaj.180066

    # Single DOI to file:
    python -m fetch_pubmed 10.1503/cmaj.180066 --out code/abstract.txt

    # Batch from a file (one DOI per line, '#' lines ignored):
    python -m fetch_pubmed --list dois.txt --out-dir abstracts/

    # Batch from stdin:
    cat dois.txt | python -m fetch_pubmed --list -

    # Pipe directly into an inspector:
    python -m fetch_pubmed 10.1503/cmaj.180066 | python -m extract.inspect
"""

from __future__ import annotations

# Load code/.env so NCBI_EMAIL / NCBI_API_KEY are populated before any HTTP call.
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_FALLBACK_EMAIL = "anonymous@example.com"


def _email() -> str:
    """Return NCBI_EMAIL if set, otherwise a clearly-fake address.

    NCBI's policy is to include a contact email on every request. With a
    fake address, requests still work, but you risk getting throttled
    or blocked under load. Setting NCBI_EMAIL is strongly recommended.
    """
    return os.environ.get("NCBI_EMAIL") or _FALLBACK_EMAIL


def _common_params() -> dict[str, str]:
    """Email plus API key when available. Merge into every E-utilities call."""
    params = {"email": _email()}
    key = os.environ.get("NCBI_API_KEY")
    if key:
        params["api_key"] = key
    return params


def doi_to_pmid(doi: str, timeout: int = 30) -> str | None:
    """Resolve a DOI to its first matching PubMed ID, or None on miss."""
    params = {
        "db": "pubmed",
        "term": f"{doi}[doi]",
        "retmode": "json",
        **_common_params(),
    }
    url = f"{_BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.load(resp)
    ids = data.get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


def fetch_abstract_by_pmid(pmid: str, timeout: int = 30) -> str:
    """Fetch the formatted abstract text for `pmid`."""
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "abstract",
        "retmode": "text",
        **_common_params(),
    }
    url = f"{_BASE}/efetch.fcgi?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def fetch_abstract_by_doi(doi: str, timeout: int = 30) -> tuple[str, str]:
    """Resolve DOI -> PMID -> abstract. Returns (pmid, abstract_text).

    Raises LookupError if the DOI doesn't resolve to a PubMed record.
    Network errors propagate; the caller decides whether to swallow them.
    """
    pmid = doi_to_pmid(doi, timeout=timeout)
    if pmid is None:
        raise LookupError(f"no PMID found for DOI {doi!r}")
    return pmid, fetch_abstract_by_pmid(pmid, timeout=timeout)


# Capture the standard PubMed body sections: a leading capitalized label
# followed by a colon and a space, optionally with internal slashes (e.g.,
# "BACKGROUND:", "METHODS:", "BACKGROUND/AIMS:", "INTERPRETATION:").
_BODY_HEADER = re.compile(r"^(?:[A-Z][A-Z/ ]{1,30}):\s")


def extract_body(abstract_text: str) -> str:
    """Strip the journal header and author block from a PubMed-formatted abstract.

    Keeps everything from the first structured-section header (BACKGROUND,
    METHODS, RESULTS, etc.) up to but not including the trailing metadata
    (DOI:, PMCID:, PMID:, Conflict of interest, etc.).

    For unstructured abstracts (no section headers), returns the whole
    string unchanged. This is intentional: it's better to over-include
    than to throw away the actual abstract content because heuristics
    didn't find a known header.
    """
    lines = abstract_text.splitlines()

    # Find the first line that looks like a structured section header.
    start = next(
        (i for i, line in enumerate(lines) if _BODY_HEADER.match(line)),
        None,
    )
    if start is None:
        return abstract_text.strip()

    # Cut off the trailing metadata: DOI: / PMCID: / PMID: / Conflict / Copyright.
    end = len(lines)
    for i in range(start, len(lines)):
        s = lines[i].lstrip()
        if (
            s.startswith("DOI:")
            or s.startswith("PMID:")
            or s.startswith("PMCID:")
            or s.startswith("Conflict of interest")
            or s.startswith("Copyright ")
            or s.startswith("© ")  # the © symbol
        ):
            end = i
            break

    return "\n".join(lines[start:end]).strip()


def _slugify_doi(doi: str) -> str:
    """Safe filename slug for a DOI: 10.1503/cmaj.180066 -> 10.1503_cmaj.180066."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", doi).strip("_")


def _read_doi_list(source: str) -> list[str]:
    """Read DOIs from a file or stdin ('-'). One per line, '#' comments ignored."""
    if source == "-":
        lines = sys.stdin.read().splitlines()
    else:
        lines = Path(source).read_text().splitlines()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _emit(text: str, body_only: bool) -> str:
    return extract_body(text) if body_only else text


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fetch_pubmed",
        description="Fetch PubMed abstract(s) by DOI via NCBI E-utilities.",
    )
    parser.add_argument(
        "doi",
        nargs="?",
        help="A single DOI (e.g., 10.1503/cmaj.180066). "
             "Omit if using --list for batch mode.",
    )
    parser.add_argument(
        "--list",
        dest="list_source",
        help="Path to a file with one DOI per line ('-' for stdin). Comment "
             "lines starting with '#' are ignored.",
    )
    parser.add_argument(
        "--out",
        help="In single-DOI mode, write the abstract to this file instead of stdout.",
    )
    parser.add_argument(
        "--out-dir",
        help="In batch mode, write each abstract to <out-dir>/<slug>.txt.",
    )
    parser.add_argument(
        "--body",
        action="store_true",
        help="Strip the journal header, author block, and trailing metadata; "
             "keep just the BACKGROUND/METHODS/RESULTS/etc. content.",
    )
    args = parser.parse_args()

    # Batch mode.
    if args.list_source:
        dois = _read_doi_list(args.list_source)
        out_dir = Path(args.out_dir) if args.out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        for doi in dois:
            try:
                pmid, text = fetch_abstract_by_doi(doi)
            except Exception as e:  # noqa: BLE001 - per-DOI fail-soft
                print(f"# {doi}: ERROR {type(e).__name__}: {e}", file=sys.stderr)
                continue

            emitted = _emit(text, args.body)
            if out_dir:
                path = out_dir / f"{_slugify_doi(doi)}.txt"
                path.write_text(emitted)
                print(f"{doi}  PMID={pmid}  -> {path}", file=sys.stderr)
            else:
                print(f"# DOI {doi}, PMID {pmid}")
                print(emitted)
                print()
        return

    # Single-DOI mode.
    if not args.doi:
        parser.print_help(sys.stderr)
        sys.exit(2)

    pmid, text = fetch_abstract_by_doi(args.doi)
    emitted = _emit(text, args.body)

    if args.out:
        Path(args.out).write_text(emitted)
        print(f"PMID {pmid} -> {args.out}", file=sys.stderr)
    else:
        print(emitted)


if __name__ == "__main__":
    main()
