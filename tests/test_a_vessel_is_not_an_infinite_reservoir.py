"""A capillary carries a finite amount of oxygen, and can run out.

Every vessel in this model was pinned at the arterial level along its whole
length. That is a defensible approximation while the vasculature is dense --
at the configured 60 um spacing the tissue draws 9% of what flows past -- and
it is indefensible exactly where this platform's central claim lives.

The measurement it fails against is unusually clean. Movsas et al. (Am J Clin
Oncol 2001;24(5):458-61, PMID 11586096) put an Eppendorf microelectrode into
the pathologically involved prostate of 59 men AND into each man's own
muscle: carcinoma median 2.4 mmHg, muscle 30.0 mmHg, p < 0.0001, with tissue
type the only significant predictor out of nine covariates. An internal
control in the same patient is far harder to accommodate than one number.

A constant-pO2 vessel cannot accommodate it at all, and the reason is a
conservation argument rather than a calibration gap: reaching 2.4 mmHg by
thinning the vasculature demands 137-149% oxygen extraction at EVERY
consumption rate ever fitted against pimonidazole staining (M0 = 2.4-5.2
mmHg/s over nine HNSCC xenografts, on perfused-vessel maps, with a
Michaelis-Menten sink of this same form -- Wack et al., Med Phys
2016;43(7):4113, PMID 27370131). Raising consumption cannot rescue it: it
shrinks the spacing the target needs and raises the demand per vessel
together, so the ratio hardly moves.

What resolves it is the term Krogh had and this model dropped. Blood
depletes, so tissue beside the outlet of a capillary can be anoxic while
sitting 10 um from a vessel -- which is how a tumour is hypoxic at high
microvessel density, and why microvessel density is the wrong observable to
calibrate against. The prostate literature cannot agree whether MVD even
rises in cancer (2.02x, PMID 1283896; not significant, and lower by vessel
area, PMID 23069258), and both findings can stand while oxygenation is set
by perfusion.
"""

import numpy as np
import pytest

from cognisom.core.antigen_presentation import hypoxic_factor
from cognisom.core.tissue_config import (
    OXYGEN_Q_PER_HOUR, DiffusionFieldConfig)
from cognisom.gpu.steady_diffusion import INSULATED, solve_steady_state
from cognisom.gpu.vessel_transport import (
    ARTERIAL_PO2_MMHG, CAPILLARY_VELOCITY_UM_PER_S, HB_CAPACITY_FRACTIONS,
    blood_content, capillary_flow_um3_per_s, march_blood, po2_from_content,
    saturation, solve_perfused_steady, vessel_outflux)

#: Read from the config rather than pinned; a literal is a second copy of a
#: calibrated number.
OXYGEN = DiffusionFieldConfig.oxygen()
VMAX = OXYGEN_Q_PER_HOUR / 3600.0
REF = OXYGEN.initial_value
DX = 10.0

#: PMID 11586096. The same instrument in the same 59 men.
MOVSAS_MUSCLE_MMHG = 30.0
MOVSAS_CARCINOMA_MMHG = 2.4

#: PMID 1283896: no organ-confined prostate carcinoma exceeded 156
#: microvessels/mm2. Mean spacing is 1000/sqrt(MVD) um.
PROSTATE_MVD_PER_MM2 = 156.0
PROSTATE_SPACING_UM = 1000.0 / np.sqrt(PROSTATE_MVD_PER_MM2)


def _unit_cell(spacing_um=PROSTATE_SPACING_UM, length_um=1000.0):
    """One vessel at the corner of a square lattice cell.

    Insulated faces make the symmetry plane between neighbouring vessels a
    zero-flux boundary, so a single cell IS the infinite lattice.
    """
    nx = max(3, int(round(spacing_um / DX)))
    nz = max(3, int(round(length_um / DX)))
    mask = np.zeros((nx, nx, nz), bool)
    mask[0, 0, :] = True
    uptake = np.full((nx, nx, nz), VMAX / REF)
    uptake[mask] = 0.0
    start = np.full((nx, nx, nz), ARTERIAL_PO2_MMHG / 760.0)
    return start, mask, uptake


def _perfused(velocity_um_per_s, **kw):
    start, mask, uptake = _unit_cell(**kw)
    result = solve_perfused_steady(
        start, OXYGEN.diffusion_coeff, DX, vessels=mask, uptake_k=uptake,
        uptake_km=OXYGEN.uptake_km, uptake_ref=REF, axis=2,
        flow_um3_per_s=capillary_flow_um3_per_s(
            velocity_um_per_s=velocity_um_per_s))
    return result, mask


# --------------------------------------------------------------------------
# the dissociation curve
# --------------------------------------------------------------------------

def test_the_dissociation_curve_matches_measured_physiology():
    """Severinghaus 1979 eq 1, against the three tensions everyone knows.

    doi 10.1152/jappl.1979.46.3.599, which fits the standard human curve to
    +/-0.0055 saturation. Arterial blood is ~97% saturated, mixed venous
    ~75%, and half-saturation is at 26-27 mmHg.
    """
    assert float(saturation(95.0)) == pytest.approx(0.97, abs=0.01)
    assert float(saturation(40.0)) == pytest.approx(0.75, abs=0.02)
    assert float(saturation(26.9)) == pytest.approx(0.50, abs=0.01)
    assert float(saturation(0.0)) == 0.0


def test_the_curve_is_cooperative_not_linear():
    """The sigmoid is the point.

    A linear carrier would give up oxygen in proportion to the tension drop.
    Haemoglobin gives up most of its load over a narrow band, which is why
    tissue can hold a low tension while blood still unloads -- and why this
    cannot be modelled as a linear reservoir.
    """
    p = np.array([10.0, 20.0, 30.0, 40.0, 60.0, 95.0])
    s = saturation(p)
    assert np.all(np.diff(s) > 0)
    steep = (s[3] - s[1]) / 20.0          # 20 -> 40 mmHg
    flat = (s[5] - s[4]) / 35.0           # 60 -> 95 mmHg
    assert steep > 6.0 * flat, (steep, flat)


def test_haemoglobin_dominates_the_dissolved_oxygen():
    """Roughly eighty-fold, which is the reason haemoglobin exists.

    The tissue field carries dissolved oxygen only, about 0.11 of an
    atmosphere-fraction at physioxia. Blood carries ~9.1 fraction-units
    bound. A model that fed the tissue field's own capacity into a vessel
    would understate delivery by that factor.
    """
    assert HB_CAPACITY_FRACTIONS == pytest.approx(9.08, abs=0.1)
    assert HB_CAPACITY_FRACTIONS / REF > 50.0


def test_content_inverts_exactly():
    """po2_from_content undoes blood_content.

    Severinghaus eq 2 inverts the saturation alone; what has to be inverted
    here is saturation plus the dissolved term, so this is a bisection.
    """
    p = np.array([0.0, 2.4, 10.0, 26.9, 40.0, 95.0, 200.0])
    assert np.allclose(po2_from_content(blood_content(p)), p, atol=1e-6)


# --------------------------------------------------------------------------
# the impossibility that forced this
# --------------------------------------------------------------------------

def test_a_constant_vessel_cannot_reach_the_measured_tension_legally():
    """Thinning a Dirichlet vasculature to 2.4 mmHg overdraws the blood.

    This is the defect, stated as a test. Find the spacing at which a
    constant-pO2 lattice reaches the measured carcinoma median, then ask
    what fraction of the arterial oxygen content one capillary would have to
    surrender to supply the tissue it serves. Above 1 the boundary condition
    is inventing oxygen.
    """
    def median_mmhg(spacing_um):
        n = max(3, int(round(spacing_um / DX)))
        mask = np.zeros((n, n, 1), bool)
        mask[0, 0, 0] = True
        uptake = np.full((n, n, 1), VMAX / REF)
        uptake[mask] = 0.0
        solved = solve_steady_state(
            np.full((n, n, 1), REF), OXYGEN.diffusion_coeff, DX,
            uptake_k=uptake, uptake_km=OXYGEN.uptake_km, uptake_ref=REF,
            held=mask, held_value=REF, faces=INSULATED)
        return float(np.median(solved.field[~mask])) * 760.0

    lo, hi = 20.0, 2000.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if median_mmhg(mid) > MOVSAS_CARCINOMA_MMHG else (lo, mid)
    spacing = 0.5 * (lo + hi)
    assert median_mmhg(spacing) == pytest.approx(MOVSAS_CARCINOMA_MMHG, abs=0.5)

    # Oxygen demanded by the tissue one millimetre of vessel serves, against
    # the oxygen a millimetre of capillary carries past it.
    demand = (spacing ** 2) * 1000.0 * VMAX                 # fraction um^3/s
    delivery = capillary_flow_um3_per_s() * float(blood_content(
        ARTERIAL_PO2_MMHG))
    assert demand / delivery > 1.0, (
        f"a constant vessel {spacing:.0f} um apart reaches "
        f"{MOVSAS_CARCINOMA_MMHG} mmHg while extracting only "
        f"{demand / delivery:.0%} -- if this ever passes, the oxygen budget "
        "no longer forbids the old boundary condition and this module's "
        "reason for existing needs rechecking")


def test_extraction_is_measured_and_cannot_exceed_everything_available():
    """The quantity the old boundary violated silently.

    Extraction is read from the solved field rather than assumed, so it is
    an output. Blood cannot give up more than it carries, at any perfusion,
    including the stagnant end where the field goes anoxic.
    """
    for velocity in (CAPILLARY_VELOCITY_UM_PER_S, 150.0, 48.0, 10.0):
        result, _ = _perfused(velocity)
        assert result.extraction.size > 0
        assert np.all(result.extraction <= 1.0 + 1e-9), (
            velocity, result.max_extraction)
        assert np.all(result.extraction >= 0.0), velocity


# --------------------------------------------------------------------------
# the transport
# --------------------------------------------------------------------------

def test_blood_cools_downstream_and_never_reheats():
    """Oxygen leaves the blood monotonically along a consuming tissue.

    A vessel whose tension rose downstream would be collecting oxygen from
    tissue that has none to give.
    """
    result, mask = _perfused(150.0)
    profile = result.blood_po2[0, 0, :] * 760.0
    assert profile[0] == pytest.approx(ARTERIAL_PO2_MMHG, rel=1e-6)
    assert np.all(np.diff(profile) <= 1e-9), profile[:12]
    assert profile[-1] < profile[0]


def test_the_tissue_gains_exactly_what_the_blood_loses():
    """The coupling is conservative by construction.

    The flux debited from the blood is read from the discrete Laplacian of
    the solved field rather than from an assumed wall conductance, so the
    two sides are the same number. Checked here against the tissue's own
    total consumption, which is what that flux must equal at steady state.
    """
    result, mask = _perfused(150.0)
    flux = vessel_outflux(result.field, mask, OXYGEN.diffusion_coeff, DX)
    delivered = float(flux[mask].sum())

    free = ~mask
    c = result.field[free]
    consumed = float(np.sum(VMAX / REF * REF * c / (OXYGEN.uptake_km + c))
                     * DX ** 3)
    assert delivered == pytest.approx(consumed, rel=0.02), (
        delivered, consumed)


def test_a_marching_vessel_needs_a_positive_flow():
    _, mask = _perfused(150.0)
    with pytest.raises(ValueError, match="flow must be positive"):
        march_blood(np.zeros(mask.shape), mask, flow_um3_per_s=0.0)


def test_a_varying_held_value_must_fit_the_field():
    start, mask, uptake = _unit_cell(length_um=30.0)
    with pytest.raises(ValueError, match="shaped like the field"):
        solve_steady_state(start, OXYGEN.diffusion_coeff, DX, held=mask,
                           held_value=np.zeros((2, 2, 2)))


def test_the_solver_holds_each_vessel_voxel_at_its_own_level():
    """The gradient along the vessel is the whole point.

    The held value used to collapse through ``max()``, which would pick the
    inlet and discard every voxel downstream of it -- silently turning a
    depleting capillary back into the infinite reservoir being replaced.
    """
    start, mask, uptake = _unit_cell(length_um=200.0)
    blood = np.array(start)
    ramp = np.linspace(ARTERIAL_PO2_MMHG, 5.0, mask.shape[2]) / 760.0
    blood[0, 0, :] = ramp
    solved = solve_steady_state(
        blood, OXYGEN.diffusion_coeff, DX, uptake_k=uptake,
        uptake_km=OXYGEN.uptake_km, uptake_ref=REF, held=mask,
        held_value=blood, faces=INSULATED)
    assert np.allclose(solved.field[0, 0, :], ramp, atol=1e-12)
    assert solved.field[0, 0, -1] < 0.5 * solved.field[0, 0, 0]


# --------------------------------------------------------------------------
# against the measurement
# --------------------------------------------------------------------------

@pytest.mark.parametrize("target,velocity", [
    (MOVSAS_MUSCLE_MMHG, 162.0),
    (MOVSAS_CARCINOMA_MMHG, 48.0),
])
def test_both_measured_tensions_are_reachable_at_measured_vessel_density(
        target, velocity):
    """One consumption rate, one vessel density, two measured tensions.

    The geometry is held at the measured prostate value and the consumption
    at the bottom of Wack's fitted range. Only perfusion differs, which is
    what tumour vasculature actually differs in -- and the velocities land
    at 32% and 10% of the textbook intravital 0.5 mm/s.
    """
    result, mask = _perfused(velocity)
    median = float(np.median(result.field[~mask])) * 760.0
    assert median == pytest.approx(target, abs=0.6), (velocity, median)
    assert np.all(result.extraction <= 1.0 + 1e-9)


def test_perfusion_is_what_moves_antigen_presentation():
    """The coupling this platform exists to model, now able to fire.

    At the old constant-pO2 boundary the tissue sat at 79 mmHg and the
    hypoxic factor was 0.999 -- the MHC-I mechanism was wired up, validated
    in direction against 592 TCGA tumours, and quantitatively inert. Across
    Movsas's two measured tissues it now moves more than twofold.
    """
    muscle, muscle_mask = _perfused(162.0)
    tumour, tumour_mask = _perfused(48.0)
    healthy = float(hypoxic_factor(np.median(muscle.field[~muscle_mask])))
    hypoxic = float(hypoxic_factor(np.median(tumour.field[~tumour_mask])))
    assert healthy > 0.95, healthy
    assert hypoxic < 0.55, hypoxic
    assert healthy / hypoxic > 2.0, (healthy, hypoxic)


def test_normal_perfusion_leaves_the_tissue_oxygenated():
    """The sanity end: at textbook velocity nothing is hypoxic.

    Blood should leave at a venous tension, not an arterial one, and the
    extraction should be a modest fraction rather than the whole load.
    """
    result, mask = _perfused(CAPILLARY_VELOCITY_UM_PER_S)
    outlet = float(result.blood_po2[0, 0, -1]) * 760.0
    assert 30.0 < outlet < 70.0, outlet
    assert result.max_extraction < 0.30, result.max_extraction
    tissue = result.field[~mask] * 760.0
    assert float((tissue < 10.0).mean()) == 0.0
    assert float(result.exhausted.max()) == 0.0
