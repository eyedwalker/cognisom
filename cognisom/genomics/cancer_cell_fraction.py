"""What share of the cancer cells carries a variant.

WHY THIS EXISTS. The platform ranks neoantigen candidates by predicted
MHC binding. The engine's own three-arm experiment on 2026-11-27 said
the thing that decides the outcome is not affinity but COVERAGE: a
peptide displayed by every cancer cell cleared the tumour (median 0
cells left of 10 founders), while one displayed by 30% removed exactly
its carriers and left the rest, in all five seeds. See
``validation/neoantigen_target_choice.py``.

That made coverage the load-bearing input, and the pipeline could not
compute it. ``Neoantigen.clonality`` has always been ``None``, by
design rather than by oversight, because cancer-cell fraction needs
tumour purity and local copy number and allele frequency alone does not
give it. This module is that calculation.

WHAT IT REFUSES TO DO. It never returns a point estimate. From bulk
sequencing the carrier fraction ``f`` and the number of mutant copies
per carrier ``m`` enter the data only as their product, so a single
number would be a choice dressed as a measurement. Every answer is an
interval, and when an input is missing the answer is ``None``.

THE ARITHMETIC, derived rather than cited. For a locus where a tumour
cell holds ``C`` copies and a normal cell holds ``N`` (2 on an
autosome, 1 on the single X or Y of a male), reads at that locus come
from both populations in proportion to purity ``p``:

    reads from tumour cells  ~  p * C
    reads from normal cells  ~  (1 - p) * N
    mutant reads             ~  p * f * m

so

    VAF = p * f * m / (p * C + (1 - p) * N)

Three of these are measurable -- VAF from read counts, ``C`` from an
allele-specific or total copy-number call, ``p`` from the sample's own
clonal mutations -- and rearranging gives the product directly:

    f * m = VAF * (p * C + (1 - p) * N) / p

``m`` is an integer in 1..C and is not identifiable, which bounds ``f``:

    m = 1  (one mutant copy per carrier)   ->  f = f * m      , the most cells
    m = C  (every copy mutant)             ->  f = f * m / C  , the fewest

So ``f`` lies in ``[f*m / C, min(1, f*m)]``, and when ``f*m`` exceeds 1
the variant may be in every cancer cell. A tumour-cell fraction above 1
is not an error, it is the signature of more than one mutant copy.

HOW THIS RELATES TO THE PUBLISHED METHOD. The quantity is the one
called cancer-cell fraction in the subclonal-reconstruction literature,
and the derivation above is the standard read-sampling model, not a new
one. Two primary records, both read directly on 2026-11-27:

  Dentro, Wedge & Van Loo, "Principles of Reconstructing the Subclonal
  Architecture of Cancers", Cold Spring Harb Perspect Med 2017;7(8):
  a026625, PMID 28270531. The abstract states the principles it sets
  out "include estimating the fraction of tumor cells for SNVs and
  CNAs", which is this calculation.

  Carter et al., "Absolute quantification of somatic DNA alterations in
  human cancer", Nat Biotechnol 2012;30(5):413-421, PMID 22544022. The
  ABSOLUTE method, which is where purity-and-ploidy-aware mutation
  multiplicity came from.

Neither reference is evidence that OUR numbers are right. They say the
quantity is the accepted one and the model is the accepted model. The
status below is therefore ``derived``: this is algebra over measured
inputs, and it is only as good as the purity and copy number handed in.

WHAT IT DOES NOT MODEL. Overlapping subclones at the same locus, loss
of heterozygosity treated separately from total copy number, whole-
genome doubling, and the possibility that the variant and the
amplification arose in different cells. A post-treatment biopsy also
bounds the fraction AFTER selection and says nothing about what it was
before, which is a fact about the sample and not about this code.

PRIOR ART IN THIS REPO. ``scripts/su2c_ar_clonality.py`` did this once,
by hand, for one variant in one sample, with the sample and the locus
hardcoded and the male-X case written inline. Its result is recorded in
``demo_patients.SU2C_AR_F877L_CLONALITY``. This module generalises that
arithmetic and ``tests/test_a_variants_cell_fraction_is_an_interval.py``
pins the reproduction of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

#: Where the model comes from, in the project's provenance vocabulary.
#: ``derived`` because it is algebra over measured inputs: nothing here
#: is a fitted or assumed constant.
CCF_MODEL_STATUS = "derived"

CCF_MODEL_SOURCE = (
    "read-sampling model derived in this module's docstring; the quantity "
    "is cancer-cell fraction as set out in Dentro, Wedge & Van Loo, Cold "
    "Spring Harb Perspect Med 2017;7(8):a026625, PMID 28270531 (abstract "
    "read directly), with mutation multiplicity after Carter et al., Nat "
    "Biotechnol 2012;30(5):413-421, PMID 22544022 (record read directly)"
)

#: Normal-cell copy number at an autosomal locus.
AUTOSOMAL_NORMAL_COPIES = 2

#: Normal-cell copy number on the single X or Y of a male sample. The
#: SU2C demo patient is male and AR is on the X, which is why the
#: original script carried this case inline.
HEMIZYGOUS_NORMAL_COPIES = 1

#: z for a two-sided 95% interval on a binomial proportion.
Z95 = 1.959963984540054


class PurityUnavailable(ValueError):
    """Raised when a caller demands a fraction without a purity estimate.

    Separate from the ``None`` return of :func:`cancer_cell_fraction` so
    that a caller can choose between "skip this variant" and "fail".
    """


@dataclass(frozen=True)
class CopyState:
    """Local copy number at one locus.

    total_tumour_copies
        Copies of the locus in a single tumour cell (``C``). A float
        because it comes from a segmented log-ratio, not a count.
    normal_copies
        Copies in a single normal cell (``N``): 2 on an autosome, 1 on
        the single X or Y of a male sample.
    source
        Free text saying where the number came from. Required, because
        a copy number with no provenance is the input that made the AR
        variant look like a 23% subclone.
    """

    total_tumour_copies: float
    normal_copies: int = AUTOSOMAL_NORMAL_COPIES
    source: str = ""

    def __post_init__(self) -> None:
        if self.total_tumour_copies <= 0.0:
            raise ValueError(
                "total_tumour_copies must be positive; a locus present in "
                "no tumour cell cannot carry a somatic variant "
                f"(got {self.total_tumour_copies})")
        if self.normal_copies < 0:
            raise ValueError("normal_copies cannot be negative "
                             f"(got {self.normal_copies})")
        if not self.source:
            raise ValueError(
                "CopyState.source is required: say where the copy number "
                "came from, so a downstream reader can tell a segmented "
                "call from an assumption")

    @classmethod
    def from_copy_ratio(cls, copy_ratio: float, purity: float,
                        normal_copies: int = AUTOSOMAL_NORMAL_COPIES,
                        source: str = "") -> "CopyState":
        """Build a state from an observed-over-expected coverage ratio.

        A segmented log2 ratio ``r`` gives ``copy_ratio = 2 ** r``, which
        is the locus's total coverage relative to the normal baseline and
        so mixes the two populations:

            copy_ratio = (p * C + (1 - p) * N) / N

        Inverting it recovers ``C``. This is the step the original AR
        script wrote inline as ``(ratio - (1 - purity)) / purity``, which
        is this expression with ``N = 1``.
        """
        _check_purity(purity)
        if copy_ratio <= 0.0:
            raise ValueError(f"copy_ratio must be positive (got {copy_ratio})")
        copies = (copy_ratio * normal_copies
                  - (1.0 - purity) * normal_copies) / purity
        if copies <= 0.0:
            raise ValueError(
                f"copy_ratio {copy_ratio} at purity {purity} implies "
                f"{copies:.3f} tumour copies, which is not a tumour: check "
                "that the ratio and the normal baseline agree")
        return cls(total_tumour_copies=copies, normal_copies=normal_copies,
                   source=source)


@dataclass(frozen=True)
class CancerCellFraction:
    """The share of tumour cells carrying a variant, as an interval.

    lower, upper
        The bounds on ``f``. ``upper`` reaching 1.0 means the variant
        may be in every cancer cell.
    mutant_copies_per_tumour_cell
        The identifiable product ``f * m``. Above 1.0 it is direct
        evidence of more than one mutant copy per carrier.
    total_tumour_copies
        ``C``, carried so a reader can see what divided the lower bound.
    """

    lower: float
    upper: float
    mutant_copies_per_tumour_cell: float
    total_tumour_copies: float
    vaf: float
    purity: float
    status: str = CCF_MODEL_STATUS
    source: str = CCF_MODEL_SOURCE

    @property
    def may_be_clonal(self) -> bool:
        """True when the data cannot rule out every cancer cell carrying it."""
        return self.upper >= 1.0

    @property
    def is_certainly_subclonal(self) -> bool:
        """True when even one mutant copy per carrier leaves cells without it.

        This is the question a target ranking actually needs answered,
        and it is answerable from bulk data even though ``f`` is not.
        """
        return self.upper < 1.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def as_dict(self) -> Dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "mutant_copies_per_tumour_cell":
                self.mutant_copies_per_tumour_cell,
            "total_tumour_copies": self.total_tumour_copies,
            "vaf": self.vaf,
            "purity": self.purity,
            "may_be_clonal": self.may_be_clonal,
            "is_certainly_subclonal": self.is_certainly_subclonal,
            "status": self.status,
            "source": self.source,
        }

    def scenario(self, at: str) -> float:
        """Collapse the interval to one number, naming which end it is.

        The engine's ``clonal_fractions`` takes a single fraction per
        label, so something has to choose. That choice is a SCENARIO,
        not a measurement, and this method makes the caller write down
        which end it took rather than silently taking a midpoint.

        ``at`` is "lower", "upper", or "fewest_carriers"/"most_carriers"
        as readable aliases. There is deliberately no "midpoint": the
        interval is not a distribution and its centre means nothing.
        """
        ends = {
            "lower": self.lower,
            "fewest_carriers": self.lower,
            "upper": self.upper,
            "most_carriers": self.upper,
        }
        if at not in ends:
            raise ValueError(
                f"scenario(at={at!r}) is not one of {sorted(ends)}. There is "
                "no midpoint option on purpose: the interval bounds what the "
                "data allows and its centre is not an estimate")
        return ends[at]


def _check_purity(purity: Optional[float]) -> float:
    if purity is None:
        raise PurityUnavailable(
            "cancer-cell fraction needs a tumour purity estimate; without "
            "one an allele frequency does not count cells")
    if not 0.0 < purity <= 1.0:
        raise ValueError(
            f"purity must be in (0, 1] (got {purity}); a purity of zero "
            "describes a sample with no tumour in it")
    return float(purity)


def vaf_confidence_interval(alt_count: int, depth: int,
                            z: float = Z95) -> tuple:
    """A two-sided interval on the allele frequency from read counts.

    The Wald interval on a binomial proportion, which is what the
    original AR script used (``+/- 1.96 * sqrt(p(1-p)/n)``) and is kept
    for continuity with the recorded numbers. It is poor near 0 and 1;
    callers reading a near-zero VAF should not lean on it.
    """
    if depth <= 0:
        raise ValueError(f"depth must be positive (got {depth})")
    if not 0 <= alt_count <= depth:
        raise ValueError(
            f"alt_count {alt_count} is not within a depth of {depth}")
    p = alt_count / depth
    half = z * math.sqrt(p * (1.0 - p) / depth)
    return (max(0.0, p - half), min(1.0, p + half))


def mutant_copies_per_tumour_cell(vaf: float, purity: float,
                                  copy_state: CopyState) -> float:
    """The identifiable product ``f * m``.

    This is the only quantity bulk reads determine. Everything else this
    module returns is a consequence of not being able to split it.
    """
    p = _check_purity(purity)
    if not 0.0 <= vaf <= 1.0:
        raise ValueError(f"vaf must be in [0, 1] (got {vaf})")
    locus_reads = (p * copy_state.total_tumour_copies
                   + (1.0 - p) * copy_state.normal_copies)
    return float(vaf) * locus_reads / p


def cancer_cell_fraction(
    vaf: Optional[float],
    purity: Optional[float],
    copy_state: Optional[CopyState],
    alt_count: Optional[int] = None,
    depth: Optional[int] = None,
) -> Optional[CancerCellFraction]:
    """Bound the share of cancer cells carrying a variant.

    Returns ``None`` -- never a number -- when any input needed to make
    cells out of reads is missing. That is the whole point: a variant
    with no purity behind it gets no fraction, the way
    ``Neoantigen.clonality`` has always been left empty rather than
    defaulted to "clonal".

    ``alt_count`` and ``depth``, when both given, widen the interval by
    the read-sampling error on the VAF, so a shallow call reports a
    wider range than a deep one at the same frequency.
    """
    if vaf is None or purity is None or copy_state is None:
        return None
    p = _check_purity(purity)

    lo_vaf = hi_vaf = float(vaf)
    if alt_count is not None and depth is not None:
        lo_vaf, hi_vaf = vaf_confidence_interval(alt_count, depth)

    fm_point = mutant_copies_per_tumour_cell(float(vaf), p, copy_state)
    fm_lo = mutant_copies_per_tumour_cell(lo_vaf, p, copy_state)
    fm_hi = mutant_copies_per_tumour_cell(hi_vaf, p, copy_state)

    c = copy_state.total_tumour_copies
    lower = min(1.0, fm_lo / c)
    upper = min(1.0, fm_hi)
    if upper < lower:                      # only reachable if C < 1
        lower, upper = upper, lower
    return CancerCellFraction(
        lower=lower,
        upper=upper,
        mutant_copies_per_tumour_cell=fm_point,
        total_tumour_copies=c,
        vaf=float(vaf),
        purity=p,
    )


#: A variant the data proves is in only some cancer cells: even one
#: mutant copy per carrier leaves cells without it.
CLONALITY_SUBCLONAL = "subclonal"

#: A variant the data cannot rule out being in every cancer cell. NOT
#: the same as "clonal": bulk sequencing of one sample rarely proves
#: clonality, and labelling it "clonal" is the overstatement that made
#: ``Neoantigen.clonality`` default to that string in the first place.
CLONALITY_POSSIBLY_CLONAL = "possibly clonal"


def clonality_label(fraction: Optional[CancerCellFraction]) -> Optional[str]:
    """Name what the interval settles, and nothing more."""
    if fraction is None:
        return None
    return (CLONALITY_SUBCLONAL if fraction.is_certainly_subclonal
            else CLONALITY_POSSIBLY_CLONAL)


@dataclass(frozen=True)
class SampleCopyProfile:
    """One sample's purity, and the copy state wherever we can resolve it.

    This is the input a neoantigen run needs before it can say anything
    about coverage. It is deliberately a separate object from the patient
    profile: purity and copy number are properties of a SAMPLE, and a
    patient can have several, taken at different times and under
    different treatment.

    purity
        Fraction of cells in the sequenced sample that are tumour.
        ``None`` is allowed and means no fraction can be computed, which
        is better than a guess.
    copy_states
        Per gene symbol. A gene missing from this map gets no fraction
        rather than an assumed diploid state, because assuming diploid
        at an amplified locus is exactly the error that made the demo
        patient's AR variant look like a 23% subclone.
    purity_source
        Where the purity came from. Required when purity is given.
    """

    purity: Optional[float] = None
    copy_states: Dict[str, CopyState] = None  # type: ignore[assignment]
    purity_source: str = ""

    def __post_init__(self) -> None:
        if self.copy_states is None:
            object.__setattr__(self, "copy_states", {})
        if self.purity is not None:
            _check_purity(self.purity)
            if not self.purity_source:
                raise ValueError(
                    "purity_source is required when a purity is given: a "
                    "purity with no provenance cannot be audited, and it "
                    "scales every fraction derived from it")

    @property
    def can_compute_fractions(self) -> bool:
        return self.purity is not None and bool(self.copy_states)

    def fraction_for_gene(
        self,
        gene: Optional[str],
        vaf: Optional[float],
        alt_count: Optional[int] = None,
        depth: Optional[int] = None,
    ) -> Optional[CancerCellFraction]:
        """The interval for a variant in ``gene``, or ``None``.

        ``None`` whenever purity is absent, the gene has no copy state,
        or the variant has no allele frequency. Three different reasons
        for the same answer, all of them "not enough was measured".
        """
        if gene is None or self.purity is None:
            return None
        state = self.copy_states.get(gene)
        if state is None:
            return None
        return cancer_cell_fraction(vaf, self.purity, state,
                                    alt_count=alt_count, depth=depth)


def founding_fractions(
    labelled: Dict[str, Optional[CancerCellFraction]],
    at: str,
) -> Dict[str, float]:
    """Turn intervals into the engine's ``clonal_fractions`` argument.

    ``EngineRunner(clonal_fractions={label: fraction})`` takes one number
    per mutation, so something has to collapse the interval. This makes
    that collapse explicit and uniform: ``at`` is required, it is the
    same end for every label in the call, and it is recorded by the
    caller rather than buried in a helper's default.

    A label whose fraction is ``None`` is left out entirely, so a variant
    the data could not place does not silently become a number. A label
    whose chosen end is 0.0 is also left out, because the engine refuses
    a zero founding fraction and a variant in none of the founding cells
    is not a subclone to seed.

    The result is a SCENARIO. The interval says what the sample allows;
    picking an end of it is a modelling choice, and for a post-treatment
    biopsy it is a choice about a tumour already under selection.
    """
    out: Dict[str, float] = {}
    for label, fraction in labelled.items():
        if fraction is None:
            continue
        value = fraction.scenario(at)
        if value > 0.0:
            out[label] = value
    return out


def ccf_report() -> Dict[str, object]:
    """What this module contributes, for the generated methods page."""
    return {
        "quantity": "cancer-cell fraction (share of tumour cells carrying "
                    "a variant)",
        "returns": "an interval, never a point estimate",
        "requires": ["variant allele frequency", "tumour purity",
                     "local tumour copy number"],
        "status": CCF_MODEL_STATUS,
        "source": CCF_MODEL_SOURCE,
        "not_modelled": [
            "overlapping subclones at one locus",
            "loss of heterozygosity apart from total copy number",
            "whole-genome doubling",
            "a variant and an amplification arising in different cells",
        ],
    }
