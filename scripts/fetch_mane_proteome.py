#!/usr/bin/env python3
"""Build the MANE Select protein reference for exome-wide neoantigen calling.

The bundled ``data/reference_proteins.fasta`` carries 37 curated driver genes.
Anything outside that panel has no protein sequence, so no peptide can be
generated for it -- which is why a tumour that should yield one to three
hundred candidates yields a handful. This script fetches the MANE Select
proteome (about 19,300 genes) so the rest of the exome is reachable.

Why MANE Select specifically, and why the summary file:

``vep_annotator`` pins annotation to MANE Select transcripts, so resolving
peptides against the *same* transcript makes residue numbering exact by
construction rather than by coincidence. The Ensembl protein FASTA alone is
not enough to do that: 77 genes appear twice in it, once as MANE Select and
once as MANE Plus Clinical, and the two differ in length. Keying on the gene
symbol and taking whichever record comes first silently picks the wrong
isoform for those genes. BRAF is the clearest example -- the Plus Clinical
record is 807 residues and has threonine at position 600, so the canonical
V600E hotspot would fail its wild-type check or, worse, be scored against the
wrong residue. The summary file carries a ``MANE_status`` column, so this
script uses it to keep Select records only.

Output is written gzipped next to the curated FASTA, in the same
``>GENE|ACCESSION Description`` header format the loader already parses. It is
deliberately not committed -- run this script to produce it.

Usage::

    python3 scripts/fetch_mane_proteome.py            # fetch and write
    python3 scripts/fetch_mane_proteome.py --verify   # also re-check hotspots
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

# `requests` rather than `urllib`: urllib reads the system CA store, which is
# empty on some macOS installs and fails every HTTPS fetch with a self-signed
# certificate error. requests bundles certifi, the repo already depends on it,
# and scripts/fetch_reference_proteome.py uses it for the same reason.
import requests

BASE = "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/"
OUT = Path(__file__).resolve().parent.parent / "cognisom" / "genomics" / "data" / "mane_select_proteins.faa.gz"

#: Well-documented hotspots, used to confirm the numbering is right before the
#: file is written. Each is (gene, 1-based residue, expected wild-type).
HOTSPOTS = [
    ("KRAS", 12, "G"), ("KRAS", 13, "G"), ("BRAF", 600, "V"),
    ("TP53", 175, "R"), ("TP53", 248, "R"), ("TP53", 273, "R"),
    ("EGFR", 858, "L"), ("EGFR", 790, "T"), ("PIK3CA", 545, "E"),
    ("PIK3CA", 1047, "H"), ("IDH1", 132, "R"), ("NRAS", 61, "Q"),
    ("PTEN", 130, "R"), ("KIT", 816, "D"), ("AR", 878, "T"),
]


def _listing() -> str:
    r = requests.get(BASE, timeout=60)
    r.raise_for_status()
    return r.text


def _resolve(pattern: str, listing: str) -> str:
    names = sorted(set(re.findall(pattern, listing)))
    if not names:
        raise SystemExit(f"could not find a file matching {pattern!r} at {BASE}")
    return names[-1]


def _download(name: str) -> bytes:
    url = BASE + name
    print(f"  fetching {name}", flush=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content


def _select_transcripts(summary_gz: bytes) -> dict:
    """Map Ensembl protein ID -> (gene symbol, RefSeq protein) for MANE Select."""
    out = {}
    text = gzip.decompress(summary_gz).decode("utf-8", "replace")
    for i, line in enumerate(text.splitlines()):
        if i == 0 or not line:
            continue
        f = line.split("\t")
        # symbol=3, RefSeq_prot=6, Ensembl_prot=8, MANE_status=9
        if len(f) > 9 and f[9] == "MANE Select":
            out[f[8]] = (f[3], f[6])
    return out


def _sequences(faa_gz: bytes) -> dict:
    """Map Ensembl protein ID -> sequence."""
    out = {}
    pid = None
    buf = []
    text = gzip.decompress(faa_gz).decode("utf-8", "replace")
    for line in text.splitlines():
        if line.startswith(">"):
            if pid:
                out[pid] = "".join(buf)
            pid = line[1:].split()[0]
            buf = []
        elif line.strip():
            buf.append(line.strip())
    if pid:
        out[pid] = "".join(buf)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="re-check the hotspot table after writing")
    args = ap.parse_args()

    print(f"MANE source: {BASE}")
    listing = _listing()
    faa_name = _resolve(r"MANE\.GRCh38\.v[\d.]+\.ensembl_protein\.faa\.gz", listing)
    sum_name = _resolve(r"MANE\.GRCh38\.v[\d.]+\.summary\.txt\.gz", listing)

    select = _select_transcripts(_download(sum_name))
    seqs = _sequences(_download(faa_name))
    print(f"  MANE Select transcripts: {len(select):,}")
    print(f"  protein records:         {len(seqs):,}")

    by_gene = {}
    for ens_prot, (symbol, refseq_prot) in select.items():
        seq = seqs.get(ens_prot)
        if seq and symbol:
            by_gene[symbol] = (seq, refseq_prot)
    print(f"  resolved Select proteins: {len(by_gene):,}")

    # Refuse to write a reference whose numbering does not reproduce the
    # hotspots. A silently mis-numbered proteome is worse than none: every
    # wild-type check downstream would either refuse real mutations or accept
    # mis-placed ones.
    bad = []
    for gene, pos, wt in HOTSPOTS:
        entry = by_gene.get(gene)
        if entry is None:
            bad.append(f"{gene}: absent")
            continue
        seq = entry[0]
        got = seq[pos - 1] if pos <= len(seq) else "?"
        if got != wt:
            bad.append(f"{gene} residue {pos}: got {got}, expected {wt}")
    if bad:
        print("\nREFUSING TO WRITE -- hotspot numbering does not check out:")
        for b in bad:
            print(f"  {b}")
        return 1
    print(f"  hotspot check: {len(HOTSPOTS)}/{len(HOTSPOTS)} correct")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt") as fh:
        for gene in sorted(by_gene):
            seq, refseq_prot = by_gene[gene]
            fh.write(f">{gene}|{refseq_prot} MANE Select\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")
    size_mb = OUT.stat().st_size / 1048576
    print(f"\nwrote {OUT} ({len(by_gene):,} proteins, {size_mb:.1f} MB gzipped)")

    if args.verify:
        from cognisom.genomics.gene_protein_mapper import _load_mane_proteome
        loaded = _load_mane_proteome()
        print(f"loader round-trip: {len(loaded):,} proteins")
        for gene, pos, wt in HOTSPOTS:
            p = loaded.get(gene)
            got = p.sequence[pos - 1] if p and pos <= p.length else "?"
            print(f"  {'OK ' if got == wt else 'XX '}{gene} {pos} = {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
