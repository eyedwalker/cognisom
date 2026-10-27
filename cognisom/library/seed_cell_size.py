"""How big the cells a run draws are, from the papers that measured them.

The run exporter draws seven lineages, and until this pass six of them
were drawn at a ``scale_um`` nobody measured: a prostate cancer cell at
18 um, a luminal cell at 15, a T cell at 8, an NK cell at 12, an M2
macrophage at 18. Only the macrophage traced to a paper (PMID 9400735).
``lineage_appearance`` labels every one of those an estimate, which is
honest and is also the whole of what a reader could learn from them.

This module writes the measured radius for four of the five, from
papers a verifier re-read, and records why the fifth stays a placeholder.
Radius, because radius is the number every piece of geometry multiplies
through: the renderer draws at 2 x radius, the tissue engine uses it for
volume exclusion and contact, and a wrong one propagates into every
area and volume.

FOUR PAPERS, FOUR CONVENTIONS
-----------------------------
None of these is a cell sitting in a tissue section, and none of them
measured the same thing the same way. Each value therefore says what it
is, because the number alone cannot:

* **Prostatic adenocarcinoma cell -- a circulating tumour cell.** No
  measurement of a prostate adenocarcinoma cell in a tissue section was
  found. Every candidate is a patient CTC sized from CellSearch images,
  and CTC (8-11 um) run about half the diameter of cultured prostate
  lines (Coumans' nine lines 10.9-19.0 um median by Coulter; Park's
  cultured LNCaP/DU145/C4-2/PC3 13.4 um by image analysis, 'just over
  half'). Three confirmed numbers disagree by convention more
  than by cell: Coumans 2013 corrects the cartridge-flattened footprint
  to a suspension diameter with a cylinder model calibrated to a Coulter
  pipette (10.7 um median, N = 185 patients); Park 2014 calibrates to an
  image analyzer on cultured lines (7.97 um mean, 523 CTC, 16 patients);
  Mendelaar 2021 reports the equivalent-circle diameter of the flattened
  2D footprint itself (10.3 um median, 19,117 CTC). The library draws
  and collides spheres, so the suspension diameter is the quantity, and
  Coumans is used: its calibration includes leukocytes at the CTC's own
  size where Park's is fitted on 13-26 um cultured cells and extrapolated
  down, and it is on the Coulter convention the library's other measured
  sizes share. Park is the runner-up and the 25-34% disagreement is on
  the record.

* **CD8+ Effector T cell -- a 72 h CD3/CD28 blast.** Notzel 2024 is the
  only human CD8-specific diameter seen in text: 11.3 +/- 2.2 um from 15
  cells measured in Fiji, live in PBS. It is a stimulated blast, not an
  antigen-specific CTL. Two Coulter volumes corroborate it -- a mixed
  CD4/CD8 day-5 blast at 705 fL (11.04 um derived) and a 48 h mixed
  lymphocyte blast at 0.41 pL (9.22 um derived) -- and the same table's
  resting CD8 (6.8 um) agrees with the resting lymphocyte this library
  already carries from Downey 1990 (6.05 um), which is the internal
  check that the two sources describe the same cell.

* **NK cell -- IL-2-cultured, migrating, a single mean.** Guldevall 2010
  measured primary human NK cells from live imaging at 8.3 um, and
  reports it only as a simulation input with the measurement 'data not
  shown': no n, SD, range or donor count. It is nevertheless the one
  confirmed human primary NK diameter. Two older numbers were refuted on
  re-reading: the '10-15 um' of Timonen 1981 is an identification range
  on Giemsa-stained cytocentrifuge smears, not a measurement, and the
  '180 um^3' of Yasaka 1982 was not measured in that paper, is cited
  from an earlier one, and averages a whole elutriation fraction that
  was only half large granular lymphocytes. Both NK rows (the
  ``cell_type`` archetype and the ``immune_cell``) take the value.

* **Macrophage M2 -- copied from M1.** The library's 'Macrophage' and
  'Macrophage M1' rows carry 10.6 um (21.2 um diameter) from Krombach
  1997, human ALVEOLAR macrophages by Coulter-type flow cytometry.
  Nothing measures an M1 against an M2, so the M2 row takes the same
  number with the same citation and says so: the two rows differ by
  shape (ruffled pseudopod against elongated spindle), not by size.

WHAT IS DELIBERATELY NOT FILLED IN
----------------------------------
The luminal epithelial cell. Three verified numbers exist and every one
is a HEIGHT of the columnar epithelium in a tissue section (27.89 um
median normal transition zone; 19-21 um in BPH), never a width and
never a suspension diameter. A height alone cannot give an
equivalent-sphere diameter, no PrEC or RWPE-1 suspension diameter exists
in text, and the one normal-cell volume (Santamaria 2022) is only in a
figure. So it keeps its placeholder, the renderer goes on labelling it
an estimate, and the heights are recorded in :data:`LUMINAL_HEIGHTS`
for whoever adds a height parameter.

WHAT ELSE A MEASURED RADIUS MOVES
---------------------------------
Not only the picture. ``TissueScaleConfig.resolve_radii_from_library``
takes any simulation-grade ``radius_um`` for an archetype it maps, and
the tissue-scale engine resolves in its constructor, so this pass also
resized three of that engine's archetypes: cancer_epithelial 5.0 -> 5.35
um (the CTC), t_cell 5.0 -> 5.65 um (the blast) and nk_cell 5.0 -> 4.15
um (the IL-2-cultured NK cell), beside the macrophage's 10.6 um that was
already there. Those radii are the contact range, the mechanics bucket,
daughter placement and volume exclusion -- the simulated cell, not the
drawn one. The modules engine behind the dashboard presets carries no
per-cell radius (its immune module has a detection radius and a kill
radius only), so the kill calibration of 02c2b6e is untouched.
``tests/test_tissue_cell_radius.py`` pins the new set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Other:
    """A candidate that was confirmed or refuted and not used."""

    pmid: str
    #: The number as the record states it, with its unit and whether it
    #: was derived.
    number: str
    what: str
    why_not: str

    def line(self, label: str) -> str:
        return (f"{label}: PMID:{self.pmid} {self.number} -- {self.what}. "
                f"Not used: {self.why_not}")


@dataclass(frozen=True)
class Measured:
    """One paper's diameter, as the paper prints it, and the convention it is in.

    Every field a rationale is required to state is a field here, so a
    value cannot be written without saying what was measured, on what,
    in what state and in what preparation.
    """

    key: str
    pmid: str
    source: str
    #: The paper's own number, in um.
    diameter_um: float
    #: How it is stated: mean or median, spread, n.
    statistic: str
    #: The same in a few words, for the rationale's first line.
    stated_as: str
    method: str
    specimen: str
    species: str
    #: Resting against activated.
    state: str
    #: Suspension against tissue section.
    preparation: str
    #: Whether the diameter includes membrane ruffles, if the paper says.
    ruffles: str
    quote: str
    where: str
    #: One line for ``entity.sourced["diameter_convention"]``.
    convention: str
    runners_up: Tuple[Other, ...] = ()
    refuted: Tuple[Other, ...] = ()
    caveat: str = ""

    @property
    def citation(self) -> str:
        return f"PMID:{self.pmid}"

    @property
    def radius_um(self) -> float:
        return self.diameter_um / 2.0

    def conditions(self) -> str:
        return (f"{self.specimen} Species: {self.species}. State: {self.state}. "
                f"Preparation: {self.preparation}. Ruffles: {self.ruffles}. "
                f"Method: {self.method} Statistic: {self.statistic}")


PROSTATE_CTC = Measured(
    key="prostate_ctc",
    pmid="23626725",
    source=("Coumans FA, van Dalum G, Beck M, Terstappen LW. Filter "
            "characteristics influencing circulating tumor cell enrichment "
            "from whole blood. PLoS One 2013;8(4):e61770 (PMC3634026): "
            "Results, 'EpCAM+CK+CD45- CTC are Smaller than Typical Cells "
            "Derived from Tumor Cell Lines'; abstract; Discussion; Table 3"),
    diameter_um=10.7,
    statistic=(
        "Median; no SD or CV is given for CTC. Table 3 gives the "
        "distribution: 84% of prostate CTC larger than 8 um, 58% > 10, "
        "34% > 12, 16% > 14, 7% > 16 um. N = 185 metastatic prostate cancer "
        "patients (IMMC-38); the number of CTC measured is not stated. The "
        "imaging CV of the calibration cell lines was 11-16%."),
    stated_as="median over N = 185 patients' CTC, CTC count not stated",
    method=(
        "Image cytometry calibrated to the Coulter principle. The CK+ area "
        "of each CTC from CellSearch image archives (automated "
        "CK+DNA+CD45- algorithm) was converted to a diameter in suspension "
        "by taking the cartridge-flattened cell as a cylinder of its imaged "
        "area A and a constant thickness D (equation 1); D = 5.2 um was "
        "least-squares fitted so that imaging medians matched Scepter "
        "Coulter-pipette medians of SKBR-3, PC3-9 and MDA-231 cells and of "
        "leukocytes, with a deviation under 0.4 um."),
    specimen=(
        "EpCAM+ CK+ DNA+ CD45- circulating tumour cells in the blood of "
        "metastatic prostate cancer patients (IMMC-38 trial, CellSearch "
        "immunomagnetic enrichment; the paper does not name the tube). A "
        "CIRCULATING cell selected by EpCAM capture, not a tumour cell in "
        "tissue: the paper itself stresses CTC are much smaller than "
        "cultured prostate cancer cells (cell lines 10.9-19.0 um median)."),
    species="human, patient-derived primary cells, not a cell line",
    state="not applicable: a circulating cell from untreated metastatic disease",
    preparation=(
        "fixed, cartridge-flattened cells imaged in the CellSearch "
        "cartridge, converted to the diameter in suspension (volume-"
        "equivalent sphere, as the Coulter principle reports); not a tissue "
        "section"),
    ruffles=("not stated; a Coulter volume and a CK+ fluorescence footprint "
             "both exclude surface detail"),
    quote=("Applying equation 1 the median diameter of CTC was determined "
           "to be 10.7 um for prostate, 11.0 um for colorectal and 13.1 um "
           "for breast CTC."),
    where=("last sentence of the Results paragraph on CTC size; the "
           "abstract repeats it ('13.1, 10.7, and 11.0 um for breast, "
           "prostate and colorectal CTC, respectively') and so does the "
           "Discussion"),
    convention=(
        "volume-equivalent suspension diameter (CellSearch image area x a "
        "5.2 um cylinder thickness, calibrated to a Coulter pipette), per "
        "PMID:23626725; a circulating tumour cell in blood, not a tissue "
        "cell; ruffle inclusion not stated"),
    runners_up=(
        Other("24416373", "7.97 +/- 1.81 um (mean; SD or SEM not stated)",
              "523 CTC from 16 chemotherapy-naive metastatic castration-"
              "resistant prostate cancer patients, CellSearch image areas "
              "converted with a factor calibrated so that cultured LNCaP, "
              "DU145, C4-2 and PC3 CellSearch areas match their suspension "
              "diameters on a CEDEX XS image analyzer; the same cultured "
              "cells read 13.38 +/- 2.54 um (Park S, Ang RR, Duffy SP, "
              "Bazov J, Chi KN, Black PC, Ma H. PLoS One 2014;9(1):e85264)",
              "it disagrees with the chosen value by 25-34%, which Park "
              "attributes to the calibration (CEDEX image analyzer against "
              "Coulter pipette) and to processing; its factor is fitted on "
              "cultured cells that read 13.38 +/- 2.54 um and extrapolated "
              "to 8 um CTC, where "
              "Coumans' fit includes leukocytes at the CTC's own size and "
              "sits on the Coulter convention the library's other measured "
              "diameters share. It does state a cell-level n, which the "
              "chosen paper does not, so it is the runner-up and not a "
              "rejection"),
        Other("32949099", "10.3 um (median 'computed diameter')",
              "19,117 CK+ single CTC from 275 cartridges of 185 Erasmus MC "
              "prostate cancer patients: sqrt(4 x 83.6 um^2 / pi) of the "
              "ACCEPT cytokeratin area, IQR 63.5 um^2; cultured LNCaP read "
              "20.7 um the same way (Mendelaar PAJ, Kraan J, Van M, Zeune "
              "LL, Terstappen LWMM, Oomen-de Hoop E, Martens JWM, Sleijfer "
              "S. Mol Oncol 2021;15(1):116-125)",
              "an equivalent-circle diameter of the PROJECTED area of a "
              "cell flattened against the cartridge, with no flattening "
              "correction: a 2D footprint, not a suspension diameter, so it "
              "is the wrong quantity for a radius however large its n. It "
              "corroborates the size scale"),
    ),
    caveat=(
        "No measurement of a prostate adenocarcinoma cell in a tissue "
        "section was found; every candidate is a CTC sized from CellSearch "
        "images, and CTC (8-11 um) run about half the diameter of cultured "
        "prostate lines: Park (PMID:24416373) puts them 'just over half' "
        "the cultured LNCaP/DU145/C4-2/PC3 cells (7.97 against 13.38 um on "
        "the CEDEX) and its Discussion quotes LNCaP at 18 um from the "
        "literature; Coumans' nine lines (PC3-9 among them) ran 10.9-19.0 "
        "um median by Coulter; Mendelaar's LNCaP footprint read 20.7 um -- "
        "none of them in a tissue either. The engine's cancer cell is an "
        "in-situ tumour cell, so this value is probably an underestimate "
        "for it. It is stored because it is the only measured human "
        "prostate cancer cell diameter with a stated suspension "
        "convention, and the conditions say what it is. Replaces a "
        "placeholder 9.0 um radius halved from an unprovenanced scale_um "
        "of 18."),
)

CD8_EFFECTOR = Measured(
    key="cd8_effector",
    pmid="39273504",
    source=("Notzel M, Mahamid M, Kronstein-Wiedemann R, Ziemssen T, Akgun K. "
            "Raman Spectroscopy of Optically Trapped Living Human T Cell "
            "Subsets and Monocytes. Int J Mol Sci 2024;25(17):9557 "
            "(PMC11394728): Appendix A, Table A1; Materials and Methods, "
            "data-analysis paragraph"),
    diameter_um=11.3,
    statistic=(
        "Mean +/- SD 11.3 +/- 2.2 um, n = 15 cells (CV about 19%). The same "
        "table: unstimulated CD8 6.8 +/- 0.6, unstimulated CD4 6.6 +/- 0.5, "
        "stimulated CD4 10.9 +/- 1.4, monocytes 8.5 +/- 0.4 um."),
    stated_as="mean +/- SD 2.2 um of 15 cells",
    method=(
        "Light-microscopy image measurement: 'The average cell diameter was "
        "estimated from 15 cells using Fiji 2.3.0 (Table A1)'. The imaging "
        "mode, and whether the cells were measured while optically trapped, "
        "are not stated; it is a side measurement in a Raman spectroscopy "
        "paper, not its endpoint."),
    specimen=(
        "Primary CD8+ T cells negatively selected from healthy-donor "
        "buffy-coat PBMC (Miltenyi, autoMACS Pro; 94% CD3+CD8+ by flow "
        "cytometry), stimulated 72 h with ImmunoCult Human CD3/CD28 T Cell "
        "Activator (STEMCELL), activation confirmed by CD25; live, unfixed, "
        "in PBS."),
    species="human, primary cells, not a cell line",
    state=(
        "ACTIVATED: a 72 h CD3/CD28 blast, which the paper calls "
        "'stimulated', not an antigen-specific effector CTL. The resting "
        "CD8 in the same table (6.8 um) agrees with the resting lymphocyte "
        "this library carries from PMID:2272970 (6.05 um)"),
    preparation=(
        "suspension: live cells in PBS settled as a monolayer in a "
        "glass-bottom dish; not fixed, not a tissue section"),
    ruffles="not stated",
    quote=("Table A1 Average cell diameter estimated from 15 cells for each "
           "cell type. Cell Type | Diameter (Mean +/- SD, um) | CD4 | 6.6 +/- "
           "0.5 | CD4 stim | 10.9 +/- 1.4 | CD8 | 6.8 +/- 0.6 | CD8 stim | "
           "11.3 +/- 2.2 | Mono | 8.5 +/- 0.4"),
    where=("Appendix A, Table A1; Methods: 'The average cell diameter was "
           "estimated from 15 cells using Fiji 2.3.0 (Table A1).'"),
    convention=(
        "image-measured diameter of live, unfixed cells in PBS (Fiji, "
        "n = 15), per PMID:39273504; a 72 h CD3/CD28-activated blast; "
        "ruffle inclusion not stated"),
    runners_up=(
        Other("32775494", "11.04 um DERIVED from 705 +/- 23 fL",
              "equivalent-sphere diameter, d = (6V/pi)^(1/3), of a "
              "Multisizer III (Coulter) mean cell volume five days after "
              "anti-CD3/CD28 Dynabead activation of a mixed CD4+/CD8+ "
              "human T-cell population in OpTmizer + 5% human serum; the "
              "paper reports fL only and never a diameter; other media gave "
              "695-719 fL and the cells rest down to 340-400 fL by day "
              "9-11 (Ghassemi S et al., June CH, O'Connor RS. Mol Ther "
              "Methods Clin Dev 2020;18:595-606)",
              "not CD8-purified, and not a diameter in the paper; it "
              "corroborates the chosen value to 2%"),
        Other("31700012", "9.22 um DERIVED from 0.41 pL",
              "equivalent-sphere diameter of a Scepter (Coulter) mean "
              "volume at 48 h of anti-CD3 + IL-2, mixed human peripheral "
              "blood lymphocytes (T 75-90%, B 5-15%, NK 5-10%), read from "
              "a representative histogram with no n or SD; 0.17 pL resting "
              "(6.87 um) and 0.39 pL at 24 h (9.06 um) (Marakhova I, "
              "Yurinskaya V, Aksenov N, Zenin V, Shatrova A, Vereninov A. "
              "Sci Rep 2019;9:16253)",
              "an earlier time point of a mixed, asynchronous lymphocyte "
              "population; it brackets the chosen value from below"),
    ),
    caveat=(
        "Coulter-derived lymphocyte diameters may run about 10% high: "
        "Segel, Cokelet & Lichtman 1981 (PMID 7214019) found an electronic "
        "lymphocyte volume of 289 um^3 against a true 210 (shape factor "
        "1.38, Isoton hyperosmolar). The chosen value is image-measured and "
        "not subject to that. Replaces a placeholder 4.0 um radius halved "
        "from an unprovenanced scale_um of 8."),
)

NK_PRIMARY = Measured(
    key="nk_primary",
    pmid="21103395",
    source=("Guldevall K, Vanherberghen B, Frisk T, Hurtig J, Christakou AE, "
            "Manneberg O, Lindstrom S, Andersson-Svahn H, Wiklund M, Onfelt "
            "B. Imaging immune surveillance of individual natural killer "
            "cells confined in microwell arrays. PLoS One 2010;5(11):e15453 "
            "(PMC2980494): Methods, 'Simulations'; Supporting Figure S1 "
            "legend"),
    diameter_um=8.3,
    statistic=(
        "A single mean, 8.3 um, with no SD, range, n or donor count: it is "
        "reported as a simulation input ('the cell was spherically shaped "
        "(diameter = 8.3 um)') whose underlying measurement is 'data not "
        "shown'. The paper's generic effector cell is 13 um, probably the "
        "YTS line, and it notes the primary NK is smaller."),
    stated_as="a single mean with no SD, range or n",
    method=(
        "Measured from time-lapse imaging of live, calcein-labelled cells "
        "migrating in silicon-glass microwells in culture medium (LSM 5 "
        "Pascal inverted confocal microscope, plan-neofluar 10x/0.3 "
        "objective, a frame every 1.5 min for 9 h). Methods: 'All mean cell "
        "sizes and mean velocities used were measured directly from imaging "
        "of the cell types used in the actual experiments.' How the "
        "diameter of a migrating, polarised cell was defined on a 10x image "
        "is not stated."),
    specimen=(
        "Polyclonal primary human NK cells from lymphocyte-enriched buffy "
        "coats of healthy donors, Ficoll-Paque then negative magnetic bead "
        "selection (Miltenyi 130-092-657), cultured in IMDM + 10% human "
        "serum + 200 U/ml IL-2 and imaged with HEK 293T targets."),
    species="human, primary cells, not a cell line",
    state=(
        "IL-2-cultured (the Figure 6 legend calls them 'IL-2 activated "
        "primary human NK cells'), not freshly resting; migrating during "
        "the measurement"),
    preparation=(
        "suspension-type: live, unfixed cells settled by gravity in medium "
        "on the glass floor of a microwell; not a stained smear and not a "
        "tissue section"),
    ruffles="not stated",
    quote=("The properties of the EC mimicked that measured for primary NK "
           "cells with at diameter of 8.3 um and migration velocity "
           "randomized around 2.3 um/min while the target cell was "
           "considered to be a stationary sphere (diameter 16 um)."),
    where=("Supporting Information, Figure S1 legend (as served in the "
           "Europe PMC full text); the same value in Methods, Simulations: "
           "'A small primary NK cell was also simulated based on "
           "experimental data of migrating NK cells together with tumor "
           "targets (data not shown). ... the cell was spherically shaped "
           "(diameter = 8.3 um)'"),
    convention=(
        "image-measured diameter of live, IL-2-cultured primary NK cells "
        "migrating in medium (10x/0.3 confocal; a single mean, measurement "
        "'data not shown'), per PMID:21103395; ruffle inclusion not stated"),
    refuted=(
        Other("6166701", "'10-15 um' (no mean; a midpoint of 12.5 was proposed)",
              "'LGL were identified as slightly larger than small and "
              "medium-sized lymphocytes (10-15 um)' on Giemsa-stained "
              "cytocentrifuge smears, with macrophages at 15-20 um "
              "(Timonen T, Ortaldo JR, Herberman RB. J Exp Med "
              "1981;153(3):569-582)",
              "a visual identification range for a differential count on "
              "dried, flattened smear cells, not a measurement: no mean, "
              "SD, n or sizing method, 'diameter' never occurs, and the "
              "parenthetical is grammatically attachable to the small "
              "lymphocytes. Its midpoint is not a number the paper states. "
              "A cytocentrifuged cell reads larger than its suspension "
              "diameter"),
        Other("7095831", "180 um^3 (a 7.0 um sphere)",
              "'180 u3' mean cell volume of elutriation fractions enriched "
              "in large lymphocytes (Yasaka T et al. Immunology 1982, "
              "PMC1555328, scanned)",
              "measured in no part of that paper: the volume is cited from "
              "Yasaka et al. 1981 (J Immunol 127:1515) and is the mean of a "
              "whole elutriation fraction that was only 47-51% large "
              "granular lymphocytes, and the paper's own NK size is a "
              "diameter on smear films ('7-10 u in diameter')"),
    ),
    caveat=(
        "The retrieved record supports about 7-8.5 um in suspension "
        "(resting to IL-2-cultured) with 10-15 um as the appearance on a "
        "smear. Not used for species or consistency: Ichise 2022 (PMID "
        "35113018) measured 10 um on MOUSE lung NK cells, and a 2025 NK-92 "
        "paper (PMID 41271739) contradicts itself twofold within one "
        "paragraph. Replaces a placeholder 6.0 um radius halved from an "
        "unprovenanced scale_um of 12, on both NK rows."),
)

BY_KEY: Dict[str, Measured] = {m.key: m for m in (PROSTATE_CTC, CD8_EFFECTOR, NK_PRIMARY)}

#: (entity name, entity type) -> the measurement that answers for it.
#: "NK cell" is two rows, the ``cell_type`` archetype and the
#: ``immune_cell``, and both are drawn, so both take the value.
APPLIES: Dict[Tuple[str, str], str] = {
    ("Prostatic adenocarcinoma cell", "cell_type"): "prostate_ctc",
    ("CD8+ Effector T cell", "immune_cell"): "cd8_effector",
    ("NK cell", "cell_type"): "nk_primary",
    ("NK cell", "immune_cell"): "nk_primary",
}

#: (entity, type) -> the row whose MEASURED radius it copies, and the
#: citation that row must carry for the copy to be allowed. Read from the
#: store at seed time rather than restated here, so the two rows cannot
#: drift: if M1 is ever re-sourced, re-running this pass moves M2 with it.
COPIED: Dict[Tuple[str, str], Tuple[str, str, str]] = {
    ("Macrophage M2", "immune_cell"): ("Macrophage M1", "immune_cell", "PMID:9400735"),
}

COPY_NOTE = (
    "THE MEASUREMENT IS OF HUMAN ALVEOLAR MACROPHAGES, not of a polarised "
    "macrophage of either kind: resident alveolar macrophages of "
    "nonsmoking healthy volunteers, n = 10, 21.2 +/- 0.3 um diameter by "
    "selective Coulter-type flow cytometric analysis of unfixed cells in "
    "suspension (Krombach et al. 1997, via BioNumbers BNID:103565). "
    "Polarization-specific size is unmeasured: no paper sizing an M1 "
    "against an M2 was found, so the M2 row carries the one measured human "
    "macrophage diameter the library has, exactly as the 'Macrophage' and "
    "'Macrophage M1' rows do, and the two polarizations differ by shape "
    "(ruffled pseudopod against elongated spindle) and not by size. "
    "Replaces a placeholder 9.0 um radius halved from an unprovenanced "
    "scale_um of 18.")

#: The one measured human macrophage diameter as a convention line, for
#: the rows that carry the number. The BioNumbers pass wrote the radius
#: with its citation and no ``diameter_convention`` (the field came
#: later), so until this pass the M2 copy was better documented than
#: the 'Macrophage M1' row it copied and the 'Macrophage' row beside it.
MACROPHAGE_CONVENTION = (
    "Coulter-type flow cytometric diameter of unfixed human alveolar "
    "macrophages in suspension (resident AM of nonsmoking healthy "
    "volunteers, n = 10, 21.2 +/- 0.3 um), per PMID:9400735 via BioNumbers "
    "BNID:103565; a healthy alveolar macrophage, not a tumour-associated "
    "or polarised one; ruffle inclusion not stated (a Coulter volume "
    "excludes surface detail)")

#: (entity, type) -> (the citation its MEASURED radius must carry, the
#: convention to write beside it). Nothing numeric is written here: a
#: convention describes a measurement, so it goes only beside a radius
#: that is one, against the paper it names.
CONVENTION_ONLY: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("Macrophage", "cell_type"): ("PMID:9400735", MACROPHAGE_CONVENTION),
    ("Macrophage M1", "immune_cell"): ("PMID:9400735", MACROPHAGE_CONVENTION),
}

#: The verified height measurements of the luminal epithelium, kept for
#: whoever adds a height parameter. None is a diameter. (pmid, um, what)
LUMINAL_HEIGHTS: Tuple[Tuple[str, float, str], ...] = (
    ("14511041", 27.89,
     "median epithelial HEIGHT of normal transition-zone acini, mean (sd "
     "2.48) over 20 autopsy prostates of adults < 30 y; per-section "
     "minimum 9.92 (1.67), maximum 54.38 (4.09) um; computerized "
     "histomorphometry of tissue sections; the layer includes the "
     "flattened basal cells (Babinski MA, Chagas MA, Costa WS, Sampaio FJ. "
     "BJU Int 2003;92(6):592-596; BPH arm 19.96 (2.20) um)"),
    ("9167769", 19.0,
     "mean luminal cell HEIGHT in BPH simple-prostatectomy sections of "
     "American men (11.7 um in Chinese men), image analysis; "
     "hyperplastic, androgen-stimulated glands, not normal prostate (Yu "
     "EY et al. Prostate 1997;31(3):175-179)"),
    ("8636309", 21.0,
     "'epithelial cell width' 21 +/- 0.7 um (SEM, n = 10) in BPH control "
     "prostatectomy tissue, morphometry; the axis of 'width' is "
     "unverified (body paywalled) and most defensibly the same radial "
     "columnar height, since it falls to 8 um with finasteride atrophy "
     "and the same endpoint is called 'epithelial height' by the same "
     "group later (Rittmaster RS, Norman RW, Thomas LN, Rowden G. J Clin "
     "Endocrinol Metab 1996;81(2):814-819)"),
)

#: entity -> why no radius is written. Recorded rather than guessed: a
#: wrong size propagates into every area and volume.
DECLINED: Dict[str, str] = {
    "Luminal epithelial cell": (
        "no confirmed DIAMETER. The three verified measurements are each a "
        "HEIGHT of the columnar epithelium in a tissue section, never a "
        "width and never a suspension diameter: "
        + "; ".join(f"PMID:{pmid} {um:g} um, {what}" for pmid, um, what in LUMINAL_HEIGHTS)
        + ". A height alone gives no equivalent-sphere diameter (the brief "
        "allowed a height x width cylinder, and no paper reported a width). "
        "Searched and not found: a PrEC / RWPE-1 / HPrEC suspension "
        "diameter (PubMed, PMC full text and Europe PMC body searches, "
        "about 70 articles grepped; every RWPE-1 hit was an IC50, a bead or "
        "a vesicle); a tissue-section luminal cell width; the one normal-"
        "cell VOLUME (Santamaria 2022, PMID 34355401, nucleator) is only in "
        "a figure whose image download was blocked. The 15 um scale_um "
        "stays an estimate and the renderer labels it so."),
}


@dataclass
class Run:
    written: List[Tuple[str, float, float]] = field(default_factory=list)
    corrected: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)
    #: Entities left placeholder on purpose, with the reason.
    declined: List[str] = field(default_factory=list)
    #: Rows that took (or already carried) a convention for a radius
    #: another pass measured; no number written.
    conventions: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{len(self.written)} radii written "
                f"({len(self.corrected)} corrected), "
                f"{len(self.conventions)} conventions, "
                f"{len(self.declined)} deliberately left placeholder, "
                f"{len(self.missing)} entities absent, "
                f"{len(self.refused)} refused")


def rationale(measured: Measured, entity_name: str) -> str:
    """Everything a reader needs to re-find the number and to know what it is."""
    lines = [
        f"{measured.diameter_um:g} um diameter ({measured.stated_as}) -> "
        f"{measured.radius_um:.3f} um radius, for {entity_name!r}.",
        f"METHOD: {measured.method}",
        f"SPECIMEN: {measured.specimen}",
        f"SPECIES: {measured.species}.",
        f"STATE (resting vs activated): {measured.state}.",
        f"PREPARATION (suspension vs tissue section): {measured.preparation}.",
        f"RUFFLE-INCLUSIVE: {measured.ruffles}.",
        f"QUOTE: \"{measured.quote}\" ({measured.where}).",
    ]
    lines += [other.line("RUNNER-UP") for other in measured.runners_up]
    lines += [other.line("REFUTED") for other in measured.refuted]
    if measured.caveat:
        lines.append(f"CAVEAT: {measured.caveat}")
    return "\n".join(lines)


def _label(name: str, entity_type: str) -> str:
    return f"{name} ({entity_type})"


def _record(run: Run, store, entity, label: str, before, after: float,
            changed_by: str) -> None:
    store.update_entity(entity, changed_by=changed_by)
    run.written.append((label, before.value if before else 0.0, after))
    if before is not None and abs(before.value - after) > 0.005:
        run.corrected.append(
            f"{label}: {before.value} -> {after:.3f} um [was {before.provenance.value}]")


def seed(store, *, dry_run: bool = False) -> Run:
    """Write a measured radius wherever one of these papers answers."""
    from .parameters import (ParameterError, ParameterValue, Provenance,
                             get_parameter, set_parameter)

    run = Run()
    for (entity_name, entity_type), key in APPLIES.items():
        label = _label(entity_name, entity_type)
        entity = store.find_entity_by_name(entity_name, entity_type)
        if entity is None:
            run.missing.append(label)
            continue
        measured = BY_KEY[key]
        before = get_parameter(entity, "radius_um")
        if dry_run:
            run.written.append((label, before.value if before else 0.0,
                                measured.radius_um))
            continue
        try:
            set_parameter(entity, "radius_um", ParameterValue(
                value=round(measured.radius_um, 3), unit="um",
                provenance=Provenance.MEASURED,
                citation=measured.citation, source=measured.source,
                conditions=measured.conditions(),
                rationale=rationale(measured, entity_name)))
        except ParameterError as exc:
            run.refused.append(f"{label}: {exc}")
            continue
        entity.sourced["diameter_convention"] = measured.convention
        _record(run, store, entity, label, before, measured.radius_um,
                "seed_cell_size")

    for (entity_name, entity_type), (from_name, from_type, citation) in COPIED.items():
        label = _label(entity_name, entity_type)
        entity = store.find_entity_by_name(entity_name, entity_type)
        donor = store.find_entity_by_name(from_name, from_type)
        if entity is None:
            run.missing.append(label)
            continue
        if donor is None:
            run.missing.append(_label(from_name, from_type))
            continue
        pv = get_parameter(donor, "radius_um")
        # Copy only what is itself a measurement against the expected
        # paper. Anything else would launder a placeholder through a row
        # that happens to be named like a sourced one.
        if pv is None or pv.provenance is not Provenance.MEASURED \
                or pv.citation != citation:
            run.refused.append(
                f"{label}: {from_name!r} carries no MEASURED radius against "
                f"{citation} to copy (found {pv.describe() if pv else 'nothing'})")
            continue
        before = get_parameter(entity, "radius_um")
        if dry_run:
            run.written.append((label, before.value if before else 0.0, pv.value))
            continue
        try:
            set_parameter(entity, "radius_um", ParameterValue(
                value=pv.value, unit=pv.unit, provenance=Provenance.MEASURED,
                citation=pv.citation, source=pv.source,
                conditions=(f"{pv.conditions}; applied to the M2 row: the "
                            f"measurement is of human alveolar macrophages "
                            f"and polarization-specific size is unmeasured"),
                rationale=(f"Copied from the library's {from_name!r} record, "
                           f"which carries the same citation: {pv.rationale}\n"
                           f"{COPY_NOTE}")))
        except ParameterError as exc:
            run.refused.append(f"{label}: {exc}")
            continue
        entity.sourced["diameter_convention"] = (
            f"{MACROPHAGE_CONVENTION}; copied from the {from_name!r} row (same "
            "number, same citation); polarization-specific size unmeasured")
        _record(run, store, entity, label, before, pv.value, "seed_cell_size")

    for (entity_name, entity_type), (citation, convention) in CONVENTION_ONLY.items():
        label = _label(entity_name, entity_type)
        entity = store.find_entity_by_name(entity_name, entity_type)
        if entity is None:
            run.missing.append(label)
            continue
        pv = get_parameter(entity, "radius_um")
        if pv is None or pv.provenance is not Provenance.MEASURED \
                or pv.citation != citation:
            run.refused.append(
                f"{label}: no MEASURED radius against {citation} for the "
                f"convention to describe (found {pv.describe() if pv else 'nothing'})")
            continue
        if (entity.sourced or {}).get("diameter_convention") == convention:
            run.conventions.append(f"{label} (already current)")
            continue
        if not dry_run:
            entity.sourced["diameter_convention"] = convention
            store.update_entity(entity, changed_by="seed_cell_size")
        run.conventions.append(label)

    for entity_name, why in DECLINED.items():
        if store.find_entity_by_name(entity_name) is not None:
            run.declined.append(f"{entity_name}: {why}")
    return run


def main(argv: Sequence[str] | None = None) -> int:     # pragma: no cover
    import argparse
    import contextlib
    import io

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from .store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    run = seed(store, dry_run=args.dry_run)
    print(run.summary())
    print("\nWRITTEN:")
    for name, before, after in run.written:
        was = f"{before:g}" if before else "none"
        print(f"   {name:<44} {was:>6} -> {after:6.3f} um")
    if run.corrected:
        print(f"\nCORRECTED ({len(run.corrected)}):")
        for line in run.corrected:
            print("   ", line)
    if run.conventions:
        print(f"\nCONVENTIONS ({len(run.conventions)}):")
        for line in run.conventions:
            print("   ", line)
    print(f"\nLEFT PLACEHOLDER ON PURPOSE ({len(run.declined)}):")
    for line in run.declined:
        print("   ", line[:200] + ("..." if len(line) > 200 else ""))
    if run.missing:
        print(f"\nABSENT ({len(run.missing)}):")
        for line in run.missing:
            print("   ", line)
    if run.refused:
        print(f"\nREFUSED ({len(run.refused)}):")
        for line in run.refused:
            print("   ", line[:200])
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
