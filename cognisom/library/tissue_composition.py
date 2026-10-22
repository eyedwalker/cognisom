"""Why ``DEFAULT_COMPOSITION`` is not sourced from a single-cell atlas.

:data:`cognisom.core.tissue_config.DEFAULT_COMPOSITION` says what
fraction of a simulated prostate is luminal, basal, fibroblast and the
rest. It is ten round numbers summing to 1.00 and it carries no
citation, so sourcing it against CZ CELLxGENE -- 87,105 normal prostate
cells, every one carrying a Cell Ontology term -- looks like the obvious
next job.

It is the wrong job, and this module is the measurement that shows why.
It is kept runnable rather than written down as a conclusion, because
the conclusion is the sort that gets re-litigated.

WHAT THE ATLAS ACTUALLY SAYS
----------------------------
Four datasets, all *normal* prostate, rolled up through the Cell
Ontology to the config's own archetypes:

    dataset         n    luminal   basal  fibro  smooth  endo  T cell
    4ed927e9   25,973      42.8%   28.1%   4.4%    7.7% 11.0%    2.5%
    53d208b0   20,503      52.0%   17.0%   2.5%    2.3%  5.9%   15.3%
    574e9f9e   18,472       7.3%   76.2%   5.5%    2.0%  8.4%    0.7%
    6725ee8e    2,449      89.0%   11.0%   0.0%    0.0%  0.0%    0.0%

Luminal spans 7.3% to 89.0% -- a 12.2-fold spread -- and basal 11.0% to
76.2%. One dataset reports a prostate that is three-quarters basal;
another reports one that is nine-tenths luminal. They are the same
organ.

The disagreement with the config is at most 2.5-fold. The disagreement
*within the evidence* is 12-fold. Fitting the config to this would be
fitting noise, and which way it moved would depend on which datasets
happened to be in the census that month.

AND IT IS NOT RANDOM NOISE
--------------------------
Splitting the same cells by how the sample was prepared:

    suspension        luminal   basal  fibro  smooth  endo  T cell
    cell (41,808)       33.9%   43.2%   3.7%    2.1%  6.8%    7.8%
    nucleus (25,973)    42.8%   28.1%   4.4%    7.7% 11.0%    2.5%

Smooth muscle is 3.7x higher in nuclei and T cells are 3.1x higher in
whole cells. That is the dissociation signature, not biology: muscle
and endothelium are held in matrix and are lost when tissue is
dissociated into cells, while lymphocytes come free easily and are
over-recovered. :mod:`cognisom.library.cellxgene_sourcing` already
says this about granulocytes -- "their absence is a property of the
assay, not of blood" -- and composition is the quantity that statement
most directly forbids.

TWO TRAPS ON THE WAY HERE
-------------------------
*The ontology graph does not carry the distinction that matters.*
``CL:0002341`` (basal cell of prostate epithelium) and ``CL:0002340``
(luminal cell of prostate epithelium) have **identical ancestor sets**:
both hang off ``CL:0002231`` (epithelial cell of prostate), and neither
descends from ``CL:0000646`` (basal cell). A pure graph rollup
therefore cannot separate the two archetypes that are 45% of the
config. The anchors below name those terms explicitly for that reason,
and a rollup that only walked ancestry scored 49.3% of cells
unassignable against 77.8% here.

*The method decides the answer.* Three rollups of the same cells
disagreed on the direction of the error, not merely its size:

    archetype       regex    graph only   graph + terms
    fibroblast     4.9x over  0.41x under   0.26x under
    smooth_muscle  3.0x over  0.67x         0.43x under
    basal          1.9x under 0.08x under   2.49x OVER

Any one of those, reported alone, reads as a finding.

WHAT WOULD ACTUALLY SOURCE IT
-----------------------------
Stereology and morphometry on intact tissue, which never dissociates
anything: Bartsch et al. (J Urol 1979) is the classic light-microscopic
study of normal human prostate, and Arenas et al. (Int J Androl) covers
281 autopsy prostates aged 20-84. Spatial transcriptomics preserves
composition for the same reason.

One conversion stands between those papers and this config, and it is
not a detail: stereology reports **volume** fractions and
``DEFAULT_COMPOSITION`` is a **number** fraction, used to decide how
many cells of each kind to place. Converting one to the other needs a
per-archetype cell volume, and a luminal secretory cell and a
lymphocyte differ by more than an order of magnitude in volume -- so
reading a published volume fraction straight into this dict would be
its own error, in the direction of far too many big cells.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: Archetype -> the Cell Ontology terms that identify it.
#:
#: Most are a single term relied on through its descendants, which is
#: what makes "skin fibroblast" (a real label on 1,139 prostate cells)
#: roll up to ``fibroblast`` correctly. The two prostate epithelial
#: entries are different: they name the terms themselves, because CL
#: files luminal and basal prostate epithelium as siblings with the
#: same ancestors and nothing in the graph tells them apart.
ARCHETYPE_ANCHORS: Dict[str, Tuple[str, ...]] = {
    "luminal_secretory": ("CL:0002340",),
    "basal": ("CL:0002341", "CL:0000646"),
    "fibroblast": ("CL:0000057",),
    "smooth_muscle": ("CL:0000192",),
    "endothelial": ("CL:0000115",),
    "t_cell": ("CL:0000084",),
    "macrophage": ("CL:0000235",),
    "nk_cell": ("CL:0000623",),
    "neuroendocrine": ("CL:0000165",),
    "cancer_epithelial": ("CL:0001064",),
}

#: The largest terms seen in normal prostate that no archetype claims,
#: with why. Not the full list -- 27 terms go unassigned, these are the
#: ones carrying essentially all of the 22.2%. Kept because
#: "unassigned" is a number worth being able to explain, and most of it
#: is genuine ambiguity rather than a gap in the table: two of these are
#: the PARENT of an archetype, which no amount of graph-walking can
#: push downward into its children.
UNASSIGNABLE: Dict[str, str] = {
    "CL:1000296": "epithelial cell of urethra -- urethral, not prostate parenchyma",
    "CL:0002231": "epithelial cell of prostate -- the PARENT of luminal and basal, so it cannot be split",
    "CL:0000034": "stem cell -- no archetype",
    "CL:0000163": "endocrine cell -- the parent of neuroendocrine, too broad to claim",
    "CL:0000151": "secretory cell -- generic, not necessarily prostate luminal",
    "CL:0000066": "epithelial cell -- too broad",
    "CL:0000775": "neutrophil -- real, but the config has no granulocyte archetype",
    "CL:0000669": "pericyte -- mural, no archetype",
    "CL:0000183": "contractile cell -- the PARENT of smooth muscle, not a descendant",
    "CL:0000451": "dendritic cell -- no archetype",
    "CL:0000236": "B cell -- no archetype",
    "CL:0002573": "Schwann cell -- no archetype",
}

OLS_ANCESTORS = ("https://www.ebi.ac.uk/ols4/api/ontologies/cl/terms/"
                 "{iri}/hierarchicalAncestors?size=500")

#: The spread above which a source is not usable for this quantity. The
#: config's worst disagreement with the atlas is 2.5x; between-dataset
#: spread is 12x. A source has to beat the thing it is correcting.
USABLE_SPREAD = 2.0


@dataclass
class Composition:
    """One measurement of what a tissue is made of."""

    label: str
    cells: int = 0
    fractions: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        top = sorted(self.fractions.items(), key=lambda kv: -kv[1])[:4]
        return (f"{self.label}: {self.cells:,} cells, "
                + ", ".join(f"{k} {100 * v:.1f}%" for k, v in top))


@dataclass
class Verdict:
    """Whether the atlas can source composition. It cannot."""

    per_dataset: List[Composition] = field(default_factory=list)
    per_suspension: List[Composition] = field(default_factory=list)
    spread: Dict[str, float] = field(default_factory=dict)
    assigned_fraction: float = 0.0

    @property
    def usable(self) -> bool:
        """True only if every archetype agrees across datasets."""
        return bool(self.spread) and max(self.spread.values()) < USABLE_SPREAD

    def describe(self) -> str:
        if not self.spread:
            return "no measurement"
        worst = max(self.spread.items(), key=lambda kv: kv[1])
        return (f"{len(self.per_dataset)} datasets, "
                f"{100 * self.assigned_fraction:.1f}% of cells assignable, "
                f"worst spread {worst[0]} {worst[1]:.1f}x -- "
                + ("usable" if self.usable else "NOT usable for composition"))


def ancestors(curies: Sequence[str], *, session=None) -> Dict[str, List[str]]:
    """Hierarchical ancestors per CL term, from EBI OLS4.

    Network. ``hierarchicalAncestors`` excludes the term itself, so
    callers that want "is this term, or below it" must add it back --
    :func:`assign_archetype` does.
    """
    import urllib.parse

    import requests

    session = session or requests.Session()
    out: Dict[str, List[str]] = {}
    for curie in curies:
        iri = "http://purl.obolibrary.org/obo/" + curie.replace(":", "_")
        enc = urllib.parse.quote(urllib.parse.quote(iri, safe=""), safe="")
        try:
            response = session.get(OLS_ANCESTORS.format(iri=enc), timeout=60,
                                   headers={"Accept": "application/json"})
            response.raise_for_status()
            terms = response.json().get("_embedded", {}).get("terms", [])
            out[curie] = sorted({t["obo_id"] for t in terms if t.get("obo_id")})
        except Exception as exc:                      # pragma: no cover
            log.warning("no ancestors for %s: %s", curie, exc)
            out[curie] = []
    return out


def assign_archetype(curie: str, ancestry: Dict[str, List[str]]
                     ) -> Optional[str]:
    """The archetype a CL term belongs to, or ``None``.

    ``None`` rather than a nearest guess, for the same reason
    :func:`cognisom.biousd.ontology.curie_for` returns ``None``: an
    unassigned cell is a number you can report, and a misassigned one
    silently moves the composition it was measured to check.
    """
    up = set(ancestry.get(curie, ())) | {curie}
    hits = [a for a, ids in ARCHETYPE_ANCHORS.items() if up.intersection(ids)]
    if len(hits) <= 1:
        return hits[0] if hits else None
    # Several matched, so prefer the one no other hit sits above.
    specific = [a for a in hits
                if not any(set(ARCHETYPE_ANCHORS[a]).intersection(
                    ancestry.get(other_id, ()))
                    for b in hits if b != a
                    for other_id in ARCHETYPE_ANCHORS[b])]
    return specific[0] if specific else sorted(hits)[0]


def spread_of(per_group: Sequence[Composition]) -> Dict[str, float]:
    """Max/min of each archetype's fraction across groups.

    ``inf`` where a group reports zero, which is the honest reading: a
    dataset that finds no macrophages at all does not agree with one
    that finds 5%.
    """
    keys = {k for c in per_group for k in c.fractions}
    out: Dict[str, float] = {}
    for key in keys:
        values = [c.fractions.get(key, 0.0) for c in per_group]
        low, high = min(values), max(values)
        out[key] = (high / low) if low > 1e-4 else float("inf")
    return out


def measure(*, census=None, min_cells: int = 1500) -> Verdict:
    """Run the whole argument against the live census.

    Network and a few seconds. Returns the per-dataset and
    per-suspension compositions and the spread between them, so the
    claim in this module's docstring can be re-checked rather than
    believed.
    """
    import cellxgene_census

    from cognisom.library.cellxgene_sourcing import CENSUS_VERSION

    opened = census is None
    census = census or cellxgene_census.open_soma(
        census_version=CENSUS_VERSION)
    try:
        obs = census["census_data"]["homo_sapiens"].obs.read(
            value_filter=("tissue_general == 'prostate gland' and "
                          "is_primary_data == True and disease == 'normal'"),
            column_names=["cell_type_ontology_term_id", "dataset_id",
                          "suspension_type"],
        ).concat().to_pandas()
    finally:
        if opened:
            census.close()

    terms = sorted(obs.cell_type_ontology_term_id.unique())
    ancestry = ancestors(terms)
    mapping = {t: assign_archetype(t, ancestry) for t in terms}
    obs["archetype"] = obs.cell_type_ontology_term_id.map(mapping)

    verdict = Verdict()
    verdict.assigned_fraction = float(obs.archetype.notna().mean())
    assigned = obs[obs.archetype.notna()]

    for column, into in (("dataset_id", verdict.per_dataset),
                         ("suspension_type", verdict.per_suspension)):
        for key, group in assigned.groupby(column, observed=True):
            if len(group) < min_cells:
                continue
            fractions = group.archetype.value_counts(normalize=True)
            into.append(Composition(
                label=str(key)[:8], cells=len(group),
                fractions={k: float(v) for k, v in fractions.items()}))

    verdict.spread = spread_of(verdict.per_dataset)
    return verdict


def main(argv: Sequence[str] | None = None) -> int:   # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    verdict = measure()
    for composition in verdict.per_dataset:
        print(" ", composition.describe())
    print("\n ", verdict.describe())
    for name, ratio in sorted(verdict.spread.items(), key=lambda kv: -kv[1]):
        print(f"    {name:20s} {ratio:6.1f}x")
    print("\n", json.dumps({"usable": verdict.usable}, indent=1))
    return 0 if verdict.usable else 1


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())
