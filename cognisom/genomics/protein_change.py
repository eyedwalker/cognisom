"""One parser for an HGVS protein change, because three disagreed.

A protein change arrives as a short string -- ``p.N218T``,
``p.Asn218Thr``, ``p.Y429*``, ``p.A1679Pfs*38`` -- and deciding whether
it is a single-residue missense substitution is the gate on the whole
neoantigen path. Three places decided it independently:

* ``dashboard/patient_engine.parse_missense`` anchored the pattern at
  both ends and handled three-letter codes. Correct.
* ``genomics/neoantigen_predictor`` used
  ``re.match(r"([A-Z])(\\d+)([A-Z])", ...)`` -- **unanchored at the
  end**, so ``A1679Pfs*38`` matched as ``A1679P`` and a FRAMESHIFT was
  cut into peptides as if it were one substituted residue.
* ``genomics/mutation_adapter`` has its own, for the curated
  patent-pipeline path, which rejects on a separate hotspot whitelist.

The unanchored one was not a theoretical problem. On the SU2C prostate
patient it turned three frameshifts (KMT2C ``A1679Pfs*38``, APC
``T1556Nfs*3``, CASZ1 ``T1125Kfs*144``) into missense calls, and two of
the resulting fabricated peptides scored among that patient's six
strongest binders -- 34.2 nM and 36.6 nM on a 9-mer the tumour does not
make, because a frameshift at 1679 replaces everything downstream rather
than one residue. Meanwhile ``patient_engine`` listed the same three
variants as skipped. The two halves of the same demo disagreed about
what the tumour presents.

So the parser lives here, in ``genomics`` where variants live, and the
dashboard imports it rather than the other way round.

A frameshift IS a legitimate source of neoantigens -- often a rich one,
since the novel reading frame is entirely foreign. It needs the
downstream translated sequence, which this repository does not derive,
so the honest output is a refusal recorded as "frameshift", not a
missense peptide built from the one residue the HGVS string happens to
name first.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

#: Three-letter amino-acid codes, as VEP's HGVSp writes them.
AA3_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Sec": "U",
}

_AA = r"(?:[A-Z][a-z]{2}|[A-Z*])"

#: Anchored at BOTH ends. The ``$`` is the whole point: without it every
#: ``fs``, ``del``, ``ins``, ``dup`` and ``ext`` suffix is silently
#: discarded and the variant is mistaken for a substitution.
PROTEIN_CHANGE = re.compile(rf"^(?:p\.)?\(?({_AA})(\d+)({_AA})\)?$")


def one_letter(token: str) -> str:
    return AA3_TO_ONE.get(token, token) if len(token) == 3 else token


def parse_missense(protein_change: Optional[str]) -> Optional[Tuple[str, int, str]]:
    """``"p.N218T"`` or ``"p.Asn218Thr"`` -> ``("N", 218, "T")``.

    Returns None for anything that is not a single-residue missense
    substitution: nonsense (``*``), frameshift, in-frame indel,
    synonymous (same residue both sides) or unparseable.
    """
    if not protein_change:
        return None
    match = PROTEIN_CHANGE.match(protein_change.strip())
    if not match:
        return None
    wild, position, mutant = (one_letter(match.group(1)), int(match.group(2)),
                              one_letter(match.group(3)))
    # ``\d+`` admits "0", which is not a residue number under any
    # convention, and a positive numbering offset downstream would
    # otherwise rescue it into a real index.
    if "*" in (wild, mutant) or wild == mutant or position < 1:
        return None
    return wild, position, mutant
