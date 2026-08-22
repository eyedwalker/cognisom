"""Fetch canonical human Swiss-Prot sequences for the driver gene set."""
import sys, time, urllib.parse
import requests

GENES = """ALK APC AR ARID1A ATM BRAF BRCA1 BRCA2 CDH1 CDK12 CDKN2A CHEK2 EGFR
ERBB2 ERG ESR1 FBXW7 FOXA1 GATA3 IDH1 KEAP1 KMT2C KMT2D KRAS MAP3K1 MYC NF1
NOTCH1 NRAS PALB2 PIK3CA PTEN RB1 SMAD4 SPOP STK11 TP53""".split()

BASE = "https://rest.uniprot.org/uniprotkb/search"


def fetch(gene):
    q = f'gene_exact:{gene} AND organism_id:9606 AND reviewed:true'
    url = (f"{BASE}?query={urllib.parse.quote(q)}"
           "&fields=accession,gene_primary,protein_name,length,sequence"
           "&format=tsv&size=5")
    resp = requests.get(url, timeout=45)
    resp.raise_for_status()
    rows = resp.text.strip().splitlines()
    if len(rows) < 2:
        return None
    # Prefer the entry whose primary gene matches exactly.
    best = None
    for line in rows[1:]:
        acc, primary, name, length, seq = line.split("\t")
        if primary.upper() == gene.upper():
            best = (acc, primary, name, int(length), seq)
            break
        if best is None:
            best = (acc, primary, name, int(length), seq)
    return best


out = []
for g in GENES:
    try:
        rec = fetch(g)
    except Exception as e:
        print(f"  {g:8s} ERROR {e}", file=sys.stderr)
        continue
    if not rec:
        print(f"  {g:8s} NOT FOUND", file=sys.stderr)
        continue
    acc, primary, name, length, seq = rec
    assert len(seq) == length, f"{g}: length mismatch {len(seq)} vs {length}"
    out.append((g, acc, name, seq))
    print(f"  {g:8s} {acc:8s} {length:5d} aa  {name[:46]}", file=sys.stderr)
    time.sleep(0.15)

path = sys.argv[1]
with open(path, "w") as f:
    for gene, acc, name, seq in sorted(out):
        f.write(f">{gene}|{acc}|{len(seq)} {name}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i + 60] + "\n")

print(f"\nwrote {len(out)} proteins to {path}", file=sys.stderr)
