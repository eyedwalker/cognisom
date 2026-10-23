#!/usr/bin/env python3
"""
Immune Module
=============

Handles immune system: T cells, NK cells, macrophages.

Features:
- T cells (CD8+ cytotoxic) - closed-loop TCR-pMHC recognition
- NK cells (natural killer) - activating/inhibitory receptor balance:
  KIR-HLA licensing plus missing self, resolved from the patient's
  measured class-I typing when one is on record
  (engine/py/immune/nk_receptor.py)
- Macrophages - phagocytosis, gated by the PD-1 "don't eat me" signal
  (engine/py/immune/macrophage_phagocytosis.py)
- Immune surveillance and patrol
- Cancer recognition and killing
- Chemotaxis (follow gradients)

NOT modelled here: M1/M2 macrophage polarization. The feature list used
to claim it. It exists in this platform only as a transcriptomic marker
score over *input* expression data (genomics/cell_state_classifier.py);
ImmuneCell has no polarization field, nothing in the simulation reads
one, and no measured M1/M2 ratio exists for any patient -- the only
runtime path into that classifier ends at a synthetic expression
generator that plants the M2 markers it then rediscovers. Simulating
polarization is worth doing; seeding it from that number is not.
"""

import logging
import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from cognisom.core.module_base import SimulationModule
from cognisom.core.event_bus import EventTypes
from cognisom.engine.py.immune.mhc_loading import MHCPresentation
from cognisom.engine.py.immune.tcr_repertoire import (
    ExhaustionState,
    TCRMatch,
    TCRRepertoire,
)
from cognisom.engine.py.immune.tcell_kill import kill_outcome
from cognisom.engine.py.immune.tme_classifier import (
    TMEClassification,
    classify_tme as _classify_tme,
)
from cognisom.engine.py.spatial.ecm_barrier import (
    detection_attenuation,
    motility_attenuation,
)
from cognisom.engine.py.immune.sympathetic import (
    DEFAULT_MAX_SUPPRESSION,
    SympatheticState,
    sympathetic_attenuation,
    sympathetic_state,
)
from cognisom.engine.py.immune.nk_receptor import (
    DEFAULT_ACTIVATION_THRESHOLD,
    DEFAULT_MISSING_SELF_THRESHOLD,
    UNLICENSED_RESPONSIVENESS,
    NKLicensingProfile,
    UnknownAlleleError,
    genotyped_kir,
    hla_ligand_profile,
    licensing_profile,
    draw_kir_subset,
    missing_self_recognized,
    nk_clone_decision,
    nk_decision,
)
from cognisom.engine.py.immune.macrophage_phagocytosis import (
    DEFAULT_PD1_MAX_SUPPRESSION,
    phagocytosis_probability,
)
from cognisom.engine.py.immune.macrophage_polarization import (
    DEFAULT_ARGINASE_MAX_SUPPRESSION,
    DEFAULT_CYTOKINE_RADIUS_UM,
    DEFAULT_CYTOKINE_SATURATION_COUNT,
    DEFAULT_KILL_SPAN,
    DEFAULT_TAU_HOURS,
    NORMOXIC_OXYGEN,
    arginase_attenuation,
    arginase_attenuation,
    cytokine_output,
    hypoxia_signal,
    initial_polarization,
    kill_factor as macrophage_kill_factor,
    local_signal,
    polarization_label,
    polarization_step,
    polarization_target,
)
from cognisom.engine.py.immune.lactate import (  # noqa: E402
    DEFAULT_LACTATE_MAX_SUPPRESSION, lactate_attenuation,
)

#: patrol_speed is quoted in um/min; the engine steps in hours.
MINUTES_PER_HOUR = 60.0

#: Module lineage -> the entity-library row its parameters are read from.
#: The store matches names exactly and case-sensitively, so 'NK_cell'
#: finds nothing without this table. The macrophage maps to the
#: unpolarized cell_type row because a simulated macrophage is seeded at
#: polarization 0.0 (M0); 'Macrophage M1' and 'Macrophage M2' are
#: polarized states. A lineage not listed here -- a 'Neutrophil' added by
#: hand -- is looked up under its own name as an immune_cell.
LINEAGE_ENTITY = {
    'T_cell': ('CD8+ Effector T cell', 'immune_cell'),
    'NK_cell': ('NK cell', 'immune_cell'),
    'macrophage': ('Macrophage', 'cell_type'),
}


#: Contact durations in minutes, for a lineage the library has no
#: simulation-grade value for. ILLUSTRATIVE: chosen, not measured. The
#: library supplies both T-cell contacts (75 and 8.5 min) and both NK
#: contacts (19 and 29 min); the MACROPHAGE falls back to these, and that
#: is a gap in the literature rather than an unsearched one. A macrophage
#: does not kill a tumour cell by one timed engulfment: it nibbles, and
#: whole-cell clearance takes hours of repeated trogocytosis (~25% of
#: targets phagocytosed in 3 h at an effector:target ratio of 10:1,
#: PMID:27226489; macrophage clusters need ~24 h on a tumouroid,
#: PMID:37095318). The engulfment times that ARE measured per event are
#: for beads and red cells (under 20 min, PMID:39453740), not tumour
#: cells. So no contact-to-death time comparable to the CTL and NK ones
#: was found, and none is invented here. See ImmuneModule._engage.
DEFAULT_LETHAL_CONTACT_MIN = 60.0
DEFAULT_NONLETHAL_CONTACT_MIN = 15.0
CONTACT_FIELDS = ('lethal_contact_minutes', 'nonlethal_contact_minutes')


def chase_displacement(direction: np.ndarray, distance: float,
                       speed_um_per_min: float,
                       dt_hours: float) -> np.ndarray:
    """One chase step toward a target ``distance`` um away along ``direction``.

    ``direction`` is a unit vector. Speed is um/min and the engine steps in
    hours, so the step is speed x 60 x dt -- the conversion _patrol uses.
    Both chase sites (here and the GPU-patched update in gpu/orchestrator)
    multiplied by 0.01 instead, which is 6,000x too slow: at the default
    10 um/min and dt = 0.01 h a chasing cell moved 0.001 um per step instead
    of 6 um. A cell that recognised a target from outside the 5 um kill
    radius therefore sat frozen short of it for the rest of the run, and
    only targets first seen from inside the kill radius could ever die.

    The step is capped at the remaining distance. Once it is real a cell
    must close on its target, not fly past it -- at dt = 0.05 h the uncapped
    step is 30 um, and the cell would oscillate across its target, inside
    the kill radius only by chance.
    """
    step = speed_um_per_min * MINUTES_PER_HOUR * dt_hours
    return direction * min(step, distance)


logger = logging.getLogger(__name__)


@dataclass
class ImmuneCell:
    """State of an immune cell"""
    cell_id: int
    position: np.ndarray
    cell_type: str  # 'T_cell', 'NK_cell', 'macrophage'
    #: Lytic granules this cell is loaded with, read from the library at
    #: creation. None means the library has no sourced count for the
    #: lineage, which is the answer for a macrophage and a helper T cell
    #: and is NOT the same as zero -- zero would say the cell has
    #: degranulated, and a reserve that reaches zero would contradict the
    #: measurement that a killing cell sheds a few and stays loaded.
    lytic_granules: Optional[float] = None
    velocity: np.ndarray = None
    
    # State
    activated: bool = False
    target_cell_id: int = None
    in_blood: bool = False

    # TCR-pMHC match that activated this T cell (None for NK/macrophage
    # or for not-yet-activated T cells). Carried so _kill_target can
    # parametrize kill probability with the actual affinity that drove
    # recognition.
    active_tcr_match: Optional[TCRMatch] = None

    # Inhibitory-KIR identity of this clone, as a receptor name
    # ("KIR2DL1", ...). Drawn once at creation from the patient's NK
    # compartment and fixed for life, which is how a real NK clone's
    # receptor repertoire works. None for T cells, macrophages, and any
    # NK cell created with no patient class-I typing on record.
    kir_subset: Optional[str] = None

    # M1/M2 polarization axis, for macrophages, in [-1, +1] with
    # -1 = M1 and +1 = M2 -- the same sign convention as
    # genomics/cell2sentence._score_polarization_markers ("score:
    # -1 = pure M1, +1 = pure M2"), so a simulated macrophage and a
    # classified one are described on the same scale. 0.0 (M0,
    # unpolarized) for every other lineage and for a macrophage whose
    # polarization has not been seeded; at 0.0 the polarization-
    # dependent kill factor is exactly 1.0, so an unseeded macrophage
    # behaves exactly as it did before this field existed. See
    # engine/py/immune/macrophage_polarization.py.
    polarization: float = 0.0

    # Parameters
    speed: float = 10.0  # μm/min
    detection_radius: float = 10.0  # μm
    kill_radius: float = 5.0  # μm

    # The contact in progress, if any: when it resolves (module clock,
    # hours) and whether it ends in the target's death. Both are decided
    # once, when the contact begins; see ImmuneModule._engage.
    contact_ends_at: Optional[float] = None
    contact_lethal: bool = False

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3)



def _or_default(value, default):
    """`value` unless it is None. dict.get's default does not cover a stored None."""
    return default if value is None else value


class ImmuneModule(SimulationModule):
    """
    Immune system simulation module
    
    Manages:
    - T cells (CD8+ cytotoxic)
    - NK cells (natural killer)
    - Macrophages
    - Immune surveillance
    - Cancer killing
    
    Events Emitted:
    - IMMUNE_ACTIVATED: When immune cell activates
    - CANCER_KILLED: When cancer cell is killed
    - IMMUNE_RECRUITED: When new immune cell arrives
    
    Events Subscribed:
    - CELL_TRANSFORMED: Respond to new cancer
    - CELL_DIVIDED: Track cell population
    - EXOSOME_RELEASED: Detect danger signals
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)

        # Immune cell population
        self.immune_cells: Dict[int, ImmuneCell] = {}
        self.next_immune_id = 0

        # Parameters
        #
        # These three may arrive as an explicit None, not merely absent:
        # twin_config._scale_count returns None when the immune fraction was
        # never measured, deliberately, so that an unmeasured tumour is
        # distinguishable from a minimally infiltrated one. dict.get returns
        # that stored None rather than the default, so `range(self.n_t_cells)`
        # raised TypeError on every twin built by to_simulation_params from an
        # unmeasured profile -- which is every production path. The seeding
        # count falls back to the default; the fact that it was unmeasured is
        # carried separately by self.immune_profile_measured, so nothing is
        # lost by seeding a plausible population here.
        self.n_t_cells = _or_default(self.config.get('n_t_cells'), 15)
        self.n_nk_cells = _or_default(self.config.get('n_nk_cells'), 10)
        self.n_macrophages = _or_default(self.config.get('n_macrophages'), 8)
        self.patrol_speed = self.config.get('patrol_speed', 5.0)  # μm/min
        # Chemotaxis: the patrol's random direction is mixed with the
        # unit gradient of a spatial field, weight `chemotaxis_bias`
        # (0 = pure random walk, 1 = straight up the gradient). The SPEED
        # is untouched -- it stays the sourced patrol speed -- so this
        # adds no invented velocity; it only aims the one we have. The
        # field is the secreted chemokine MorphogenModule supplies.
        # 0.5 is an ESTIMATE of the chemotactic index; it is not a
        # measured number and the module reports how often it acted.
        self.chemotaxis_field = str(self.config.get('chemotaxis_field', 'cytokine'))
        self.chemotaxis_bias = float(self.config.get('chemotaxis_bias', 0.5))
        self.chemotaxis_steps = 0
        self.spatial_module = None
        # Tumour lactic acid blunts T and NK killing (Fischer 2007, Brand
        # 2016); read off the lactate field at the effector's position.
        self.lactate_max_suppression = float(
            self.config.get('lactate_max_suppression', DEFAULT_LACTATE_MAX_SUPPRESSION))
        # IFN-gamma secretion into the spatial field. Activated CD8 T
        # cells and NK cells are the sources (Brand 2016 measures both
        # arms by their IFN-gamma output); macrophages are not. The rate
        # is per cell per hour in the field's own [0, 1] units and is an
        # ESTIMATE: no per-cell secretion rate was sourced, and only the
        # RATIO of secretion to consumption sets the field's shape.
        self.ifn_gamma_secretion_per_h = float(
            self.config.get('ifn_gamma_secretion_per_h', 1.0))
        self.ifn_gamma_sources = 0

        # Per-lineage patrol speed, um/min. patrol_speed above is the
        # speed of any lineage with nothing more specific. Precedence: an
        # explicit 'lineage_patrol_speed' entry (a calibration sweep must
        # be able to pin a speed), then a simulation-grade value from the
        # entity library, then patrol_speed. The library is read for this
        # one field only. The same rows carry detection radius, kill
        # probability and exhaustion rate, but those are model constructs
        # this module calibrates, and they stay on config. Two other
        # speeds remain hardcoded and are NOT governed by this: the chase
        # speed (ImmuneCell.speed) and the ingestion archetypes'.
        self.lineage_patrol_speed: Dict[str, float] = {
            lineage: float(speed) for lineage, speed in
            (self.config.get('lineage_patrol_speed') or {}).items()
        }
        #: lineage -> where its patrol speed came from: 'config', the
        #: library row with the value's grade and citation, or why the
        #: library had nothing usable (and patrol_speed applies).
        self.patrol_speed_source: Dict[str, str] = {
            lineage: 'config' for lineage in self.lineage_patrol_speed
        }
        self.library_params = bool(self.config.get('library_params', True))
        # Read here rather than in initialize(), which reset() re-runs.
        for lineage in LINEAGE_ENTITY:
            self._patrol_speed_for(lineage)

        # How long a contact lasts, per lineage, in minutes: lethal (contact
        # to the target's death) and non-lethal (contact to detachment).
        # Same precedence as the patrol speed: an explicit
        # 'lethal_contact_minutes' / 'nonlethal_contact_minutes' entry, then
        # a simulation-grade library value, then the illustrative default.
        self._contact_minutes: Dict[Tuple[str, str], float] = {}
        self.contact_time_source: Dict[str, Dict[str, str]] = {}
        for lineage in LINEAGE_ENTITY:
            for lethal in (True, False):
                self._contact_minutes_for(lineage, lethal)
        #: Hours of simulated time this module has stepped through. Contacts
        #: resolve against it.
        self.clock_h = 0.0
        # Baseline per-encounter kill probability for the innate
        # lineages, and the fallback for an immune cell of an
        # unrecognized type. It is NOT a T-cell fallback: the comment
        # here used to claim it covered "a T cell engaging a cancer cell
        # with no MHC-I-displayed neoantigens", which is unreachable --
        # _recognize returns True for a T cell only together with a
        # match, and active_tcr_match is cleared only together with
        # `activated`. Instrumenting the three branches of
        # _target_kill_probability across the whole suite recorded 29
        # calls, all 29 on the T-cell Hill path and none on this one.
        # Kept as the per-lineage default (and because
        # gpu/orchestrator's immune path reads module.kill_probability),
        # so every existing config keeps working unchanged.
        self.kill_probability = self.config.get('kill_probability', 0.8)

        # ── Innate arm: per-lineage parameters ──────────────────────
        # NK cells and macrophages used to share this one flat number,
        # read at a single site. They are different cells with
        # different effector mechanisms, and validation/calibrator.py
        # has declared `nk_kill_probability` as an immune-module
        # parameter all along with nothing on this side reading it.
        self.nk_kill_probability = float(
            self.config.get('nk_kill_probability', self.kill_probability)
        )
        self.macrophage_kill_probability = float(
            self.config.get(
                'macrophage_kill_probability', self.kill_probability,
            )
        )
        # The legacy NK step rule's threshold, lifted out of the middle
        # of _recognize where it was a bare 0.4 literal. Used only when
        # no patient class-I typing is on record. It has no source.
        self.nk_missing_self_threshold = float(
            self.config.get(
                'nk_missing_self_threshold', DEFAULT_MISSING_SELF_THRESHOLD,
            )
        )
        # Deadband on the integrated NK signal. 0.0 is the pure balance
        # model: kill when activation exceeds inhibition.
        self.nk_activation_threshold = float(
            self.config.get(
                'nk_activation_threshold', DEFAULT_ACTIVATION_THRESHOLD,
            )
        )
        # PD-1 "don't eat me" gate on phagocytosis (Gordon et al.,
        # Nature 2017). 0.0 restores the previous flat behaviour.
        self.macrophage_pd1_max_suppression = float(
            self.config.get(
                'macrophage_pd1_max_suppression', DEFAULT_PD1_MAX_SUPPRESSION,
            )
        )
        # KIR-HLA receptor balance. Resolved from the patient's class-I
        # typing; stays None when there is none, which leaves the legacy
        # step rule in place rather than inventing a genotype.
        self._nk_licensing: Optional[NKLicensingProfile] = None
        # Dedicated stream for drawing each NK clone's KIR identity. It
        # is deliberately NOT numpy's global stream: drawing from that
        # would shift every downstream roll and silently perturb the
        # closed-loop T-cell test's reproducibility.
        self._nk_rng = np.random.default_rng(int(self.config.get('nk_seed', 0)))
        self._resolve_nk_licensing()

        # ── Macrophage M1/M2 polarization ──────────────────────
        # Driven each step by local IFN-gamma toward M1 and by local
        # IL-10 / TGF-beta / hypoxia toward M2. Cell-grained sampling in
        # the manner of spatial/ecm_barrier.ecm_density_at: there is no
        # diffusion solver here, and none is claimed -- the claim is the
        # gating, not the concentration field.
        self.macrophage_polarization_enabled = bool(self.config.get(
            'macrophage_polarization_enabled', True))
        self.polarization_tau_hours = float(self.config.get(
            'polarization_tau_hours', DEFAULT_TAU_HOURS))
        self.macrophage_kill_span = float(self.config.get(
            'macrophage_kill_span', DEFAULT_KILL_SPAN))
        self.cytokine_sampling_radius = float(self.config.get(
            'cytokine_sampling_radius', DEFAULT_CYTOKINE_RADIUS_UM))
        self.cytokine_saturation_count = float(self.config.get(
            'cytokine_saturation_count', DEFAULT_CYTOKINE_SATURATION_COUNT))
        # Above this polarization a macrophage stops recognizing
        # targets: an M2 macrophage is pro-tumour, not tumoricidal. The
        # default is on the far side of M0, so an unpolarized macrophage
        # still recognizes everything exactly as `return True` did.
        self.macrophage_recognition_max_polarization = float(self.config.get(
            'macrophage_recognition_max_polarization', 0.5))
        # Arginase-mediated M2 suppression of T cells, composing
        # multiplicatively with the exhaustion and sympathetic gates.
        self.arginase_max_suppression = float(self.config.get(
            'arginase_max_suppression', DEFAULT_ARGINASE_MAX_SUPPRESSION))

        # Initial macrophage polarization. None means UNMEASURED, which
        # is the honest default and exactly what
        # twin_config.to_simulation_params() emits for an unmeasured
        # immune profile. The 'm2_fraction' key that method already
        # emits is accepted here, closing plumbing that had a producer
        # and no consumer. A number supplied here is reported as
        # patient-measured ONLY when the caller also passes
        # measured=True; otherwise it is reported as a configured
        # assumption. Note that nothing in this platform measures an
        # M1/M2 ratio for any patient today: the only runtime path to
        # CellStateClassifier ends at generate_synthetic_expression(),
        # which plants the M2 markers it then rediscovers. Seeding a
        # simulated polarization from that number would walk back the
        # hardening that made twin_config report "unknown".
        m2_fraction = self.config.get(
            'macrophage_m2_fraction', self.config.get('m2_fraction', None))
        self.macrophage_m2_fraction = (
            None if m2_fraction is None else float(m2_fraction))
        self.immune_profile_measured = bool(self.config.get('measured', False))
        # 'treg_fraction' and 't_cell_initial_exhaustion' are the other
        # two keys to_simulation_params emits. They are deliberately NOT
        # consumed here: both would modify the flagship T-cell closed
        # loop, which this upgrade does not touch. They stay
        # producer-only, and that is a known gap rather than an
        # oversight.

        # TCR-pMHC matching for T cells (Upgrade 2). The repertoire is
        # patient-specific; seed is stable so the closed-loop test can
        # reproduce a recognized match.
        self.tcr_repertoire_size = int(self.config.get('tcr_repertoire_size', 1000))
        self.tcr_recognition_threshold = float(
            self.config.get('tcr_recognition_threshold', 0.7)
        )
        self.tcr_seed = int(self.config.get('tcr_seed', 0))
        # Costimulation strength on the tumor side. Real tumors
        # downregulate CD80/CD86; we leave this exposed for sensitivity
        # studies (immunotherapy ON / OFF).
        self.costimulation = float(self.config.get('costimulation', 0.6))
        self.checkpoint_block = float(self.config.get('checkpoint_block', 0.0))

        # Sympathetic / β2-adrenergic axis (Upgrade 7). Patient
        # chronic-stress proxy + propranolol-style β-blocker rescue.
        # See engine/py/immune/sympathetic.py for the rationale.
        self.stress_level = float(self.config.get('stress_level', 0.0))
        self.beta_blocker = float(self.config.get('beta_blocker', 0.0))
        self.sympathetic_max_suppression = float(
            self.config.get('sympathetic_max_suppression', DEFAULT_MAX_SUPPRESSION)
        )
        self._tcr_repertoire: Optional[TCRRepertoire] = None

        # Statistics
        self.total_kills = 0
        #: A separate stream for the granule draw, so recording the cost
        #: of a kill cannot move the simulation that produced it.
        self._granule_rng = np.random.default_rng(
            self.config.get('granule_seed', 0x6D616E))
        #: Granules shed across every kill this run. The sourced rule is
        #: that a killing cell sheds a few and stays loaded, so this
        #: growing while the reserves stay non-zero is the expected
        #: shape; reserves hitting zero would mean cells are emptying
        #: themselves, which the measurement says they do not.
        self.total_granules_released = 0
        self.total_tcell_kills = 0
        self.total_activations = 0
        self.total_recruited = 0
        self.total_tcr_recognitions = 0
        # Cumulative count of clones that have transitioned from PD-1-lo
        # precursor to PD-1-hi exhausted during this run.
        self.total_exhaustion_transitions = 0

        # Where the infiltrate starts.
        #
        # 'window' is the historical scatter and what every preset
        # before TissueSlab uses: uniform over a fixed
        # 100 x 100 x 20 um box about the domain centre, chosen for the
        # ring the cellular module seeds. On the 165 x 156 x 63 um slab
        # that window would leave the whole periphery unpatrolled.
        #
        # 'slab' scatters through the slab volume instead, sized from
        # cognisom.modules.tissue_slab.SLAB_SPEC -- the same spec the
        # cellular module fills and the vascular module perfuses.
        self.seeding = str(self.config.get('seeding', 'window'))
        self.slab_overrides = dict(self.config.get('slab', {}) or {})

        # Reference to cellular module (for target cells)
        self.cellular_module = None
    
    def initialize(self):
        """Initialize immune system"""
        print("  Creating immune cell population...")
        
        # One sampler for all three lineages, so 'slab' seeding reaches
        # every one of them. The 'window' branch keeps the original
        # three np.random.uniform draws per cell, in the original x, y, z
        # order and the original T -> NK -> macrophage cell order, so a
        # seeded run of any pre-existing preset is unchanged.
        scatter = self._slab_scatter()
        placed = 0

        def _next_position():
            nonlocal placed
            if scatter is not None:
                position = scatter[placed]
            else:
                position = [
                    np.random.uniform(50, 150),
                    np.random.uniform(50, 150),
                    np.random.uniform(40, 60),
                ]
            placed += 1
            return position

        # Create T cells (patrol tissue)
        for i in range(self.n_t_cells):
            self.add_immune_cell(
                position=_next_position(),
                cell_type='T_cell'
            )

        # Create NK cells
        for i in range(self.n_nk_cells):
            self.add_immune_cell(
                position=_next_position(),
                cell_type='NK_cell'
            )

        # Create macrophages
        for i in range(self.n_macrophages):
            self.add_immune_cell(
                position=_next_position(),
                cell_type='macrophage'
            )

        # Build the patient TCR repertoire (Upgrade 2). Same scorer the
        # closed-loop test relies on; reproducible from tcr_seed.
        self._tcr_repertoire = TCRRepertoire(
            size=self.tcr_repertoire_size,
            recognition_threshold=self.tcr_recognition_threshold,
            seed=self.tcr_seed,
        )

        # Subscribe to events
        self.subscribe(EventTypes.CELL_TRANSFORMED, self.on_cell_transformed)
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)

        print(f"    ✓ {self.n_t_cells} T cells")
        print(f"    ✓ {self.n_nk_cells} NK cells")
        print(f"    ✓ {self.n_macrophages} macrophages")
        print(f"    ✓ TCR repertoire: {self.tcr_repertoire_size} clones "
              f"(threshold={self.tcr_recognition_threshold})")
        self._announce_nk_licensing()
    
    def _slab_scatter(self):
        """Seed positions spread through the TissueSlab, or None.

        Returns an (N, 3) array covering all three lineages in seeding
        order when ``seeding == 'slab'``, and None for the historical
        window scatter.
        """
        if self.seeding != 'slab':
            return None

        from cognisom.modules.tissue_slab import (
            rng_from, scatter_in_box, slab_bounds, slab_spec,
        )

        spec = slab_spec(**self.slab_overrides)
        lo, hi = slab_bounds(
            spec['n_normal'], spec['n_cancer'],
            spacing=spec['spacing'],
            layers=spec['layers'],
            center=spec['center'],
        )
        n_total = self.n_t_cells + self.n_nk_cells + self.n_macrophages
        return scatter_in_box(
            n_total, lo, hi,
            inset=spec['immune_inset'],
            rng=rng_from(spec['seed']),
        )

    def add_immune_cell(self, position, cell_type):
        """Add immune cell"""
        cell_id = self.next_immune_id
        self.next_immune_id += 1
        
        cell = ImmuneCell(
            cell_id=cell_id,
            position=np.array(position, dtype=np.float32),
            cell_type=cell_type,
            # Seeded here rather than in initialize() so a macrophage
            # recruited mid-run starts from the same compartment mean.
            # A population fraction constrains the compartment MEAN,
            # not any individual cell; the per-cell cytokine drive then
            # diverges them.
            polarization=(
                initial_polarization(self.macrophage_m2_fraction)
                if cell_type == 'macrophage' else 0.0
            ),
        )

        # Draw this NK clone's inhibitory-KIR identity once, here, from
        # the patient's compartment. Averaging the compartment inside
        # the per-encounter roll instead would make every NK cell the
        # average NK cell: measured, that erased missing self end to
        # end, because an unlicensed subset with no cognate ligand fires
        # at any MHC-I level and so every NK cell engaged every target.
        if cell_type == 'NK_cell' and self._nk_licensing is not None:
            cell.kir_subset = draw_kir_subset(
                self._nk_licensing, self._nk_rng,
            ).receptor.value

        self._patrol_speed_for(cell_type)
        # The granules it is loaded with, from the library. None where
        # the library has no sourced count, which is the honest answer
        # for a macrophage and a helper T cell and is not zero.
        reserve = self._lytic_reserve(cell_type)
        if reserve is not None:
            cell.lytic_granules = float(reserve)
        self.immune_cells[cell_id] = cell
        return cell_id
    
    def set_spatial_module(self, spatial_module):
        """Link to the spatial module: patrol follows one of its fields,
        and activated effectors become sources of another."""
        self.spatial_module = spatial_module
        register = getattr(spatial_module, 'register_secretion', None)
        if register is not None and self.ifn_gamma_secretion_per_h > 0.0:
            register('ifn_gamma', self.ifn_gamma_sources_now)

    def ifn_gamma_sources_now(self):
        """``(position, rate)`` for every activated effector in tissue.

        Called by the spatial module after it clears the field's terms,
        so a cell that has disengaged or left stops secreting with no
        bookkeeping here. Activated CD8 T cells and NK cells only --
        Brand et al. 2016 (PMID 27641098) read both arms by their
        IFN-gamma output; a resting cell contributes nothing, which
        makes this exactly zero in a run where nothing is activated.
        """
        rate = self.ifn_gamma_secretion_per_h
        sources = []
        for cell in self.immune_cells.values():
            if cell.in_blood or not cell.activated:
                continue
            if cell.cell_type not in ('T_cell', 'NK_cell'):
                continue
            sources.append((cell.position, rate))
        self.ifn_gamma_sources = len(sources)
        return sources

    def _chemotaxis_gradient_field(self):
        if self.spatial_module is None or self.chemotaxis_bias <= 0.0:
            return None
        return (getattr(self.spatial_module, 'fields', None) or {}).get(
            self.chemotaxis_field)

    def set_cellular_module(self, cellular_module):
        """Link to cellular module for target access"""
        self.cellular_module = cellular_module

    def set_stress(
        self,
        stress_level: Optional[float] = None,
        beta_blocker: Optional[float] = None,
    ) -> SympatheticState:
        """Runtime control of the sympathetic axis.

        Update one or both knobs (``stress_level``, ``beta_blocker``)
        and return the resulting SympatheticState snapshot. Useful for
        treatment-arm experiments: e.g., run the engine, then call
        ``imm.set_stress(beta_blocker=1.0)`` mid-run to model the
        addition of propranolol.
        """
        if stress_level is not None:
            self.stress_level = float(stress_level)
        if beta_blocker is not None:
            self.beta_blocker = float(beta_blocker)
        return self.get_sympathetic_state()

    def get_sympathetic_state(self) -> SympatheticState:
        """Current SympatheticState given module configuration."""
        return sympathetic_state(
            self.stress_level,
            self.beta_blocker,
            max_suppression=self.sympathetic_max_suppression,
        )

    # ── Innate arm: NK receptor balance ─────────────────────────────

    def _resolve_nk_licensing(self) -> Optional[NKLicensingProfile]:
        """Resolve this patient's NK licensing profile, or leave it off.

        Two inputs, in order of precedence:

        ``patient_hla``
            An explicit class-I allele list. Treated as UNMEASURED
            unless ``patient_hla_measured`` is also set, so a
            hand-written panel never passes for a typing.
        ``patient_id``
            Looked up in ``hla_typer.MEASURED_HLA_TYPINGS``, which holds
            only types actually determined from a patient's reads. An id
            with no measured typing leaves the receptor model OFF and
            the legacy step rule in place. It deliberately does NOT fall
            back to a synthetic or population-default profile: an NK
            licensing prediction built on invented alleles would be
            indistinguishable from one built on real reads, and the
            whole point of this path is that it is the one place a real
            patient measurement reaches the innate simulation.

        ``kir_genes`` supplies a real KIR genotype if one exists.
        Otherwise gene content is a published carrier-frequency prior,
        flagged ``is_patient_specific=False`` -- nothing in this
        platform can type KIR (OptiType types HLA).

        An allele on neither record raises UnknownAlleleError out of the
        constructor rather than being dropped, because a dropped allele
        silently removes a ligand group and so silently unlicenses an
        NK subset.
        """
        alleles = self.config.get('patient_hla')
        measured = bool(self.config.get('patient_hla_measured', False))
        source = 'config:patient_hla'

        if not alleles:
            patient_id = self.config.get('patient_id')
            if patient_id:
                # Lazy import: the modules layer may depend on genomics,
                # but engine/py must not, and nk_receptor lives there.
                from cognisom.genomics.hla_typer import MEASURED_HLA_TYPINGS
                typing = MEASURED_HLA_TYPINGS.get(str(patient_id))
                if typing is None:
                    logger.info(
                        "no measured HLA typing for patient_id %r; the NK "
                        "receptor balance stays off and the legacy "
                        "missing-self step rule (threshold %.2f) applies",
                        patient_id, self.nk_missing_self_threshold,
                    )
                    return None
                alleles = list(typing.alleles)
                measured = True
                source = f"measured:{patient_id}:{typing.source}"

        if not alleles:
            return None

        # Fail soft on an allele the ligand map does not cover. Refusing to
        # guess a ligand group is right, but raising out of module
        # construction turns one unrecognised allele into a dead engine --
        # and this resolver runs for every patient, including demo ones.
        try:
            hla = hla_ligand_profile(
                alleles, is_patient_specific=measured, source=source,
            )
        except UnknownAlleleError as exc:
            logger.warning(
                "NK receptor balance stays OFF: %s. The legacy missing-self "
                "step rule (threshold %.2f) applies instead. Add the allele "
                "to the ligand map in engine/py/immune/nk_receptor.py to "
                "enable the receptor model for this patient.",
                exc, self.nk_missing_self_threshold,
            )
            return None

        kir_genes = self.config.get('kir_genes')
        self._nk_licensing = licensing_profile(
            hla,
            genotyped_kir(kir_genes) if kir_genes else None,
            unlicensed_responsiveness=float(self.config.get(
                'nk_unlicensed_responsiveness', UNLICENSED_RESPONSIVENESS,
            )),
        )
        return self._nk_licensing

    def get_nk_licensing_profile(self) -> Optional[NKLicensingProfile]:
        """The resolved NK licensing profile, or None if no HLA is known."""
        return self._nk_licensing

    def _announce_nk_licensing(self) -> None:
        """Report the resolved licensing profile once, at initialization.

        Emits NK_LICENSING_RESOLVED only when a typing exists. The
        absence of the event is itself the signal that the innate arm is
        running the legacy rule.
        """
        if self._nk_licensing is None:
            print(f"    - NK receptor balance: OFF (no patient class-I "
                  f"typing on record); legacy missing-self step at "
                  f"{self.nk_missing_self_threshold}")
            return
        prof = self._nk_licensing
        groups = ', '.join(g.value for g in prof.hla.groups) or 'none'
        print(f"    ✓ NK KIR-HLA balance: ligand groups [{groups}], "
              f"{prof.licensed_fraction:.0%} of the compartment licensed "
              f"(HLA {'measured' if prof.hla.is_patient_specific else 'unmeasured'}, "
              f"KIR {'genotyped' if prof.kir.is_patient_specific else 'population prior'})")
        self.emit_event(EventTypes.NK_LICENSING_RESOLVED, prof.as_dict())

    def _nk_decision(self, immune_cell, target_cell):
        """Decompose one NK-target encounter under the receptor balance.

        Per CLONE when the NK agent carries a drawn KIR identity, which
        every NK cell created through add_immune_cell does while a
        licensing profile is resolved. Falls back to the compartment
        expectation for an agent built outside that path.

        Activating signal runs on the target's per-cell mutation count,
        which the simulation actually computes; MICA/MICB/ULBP induction
        by the ATM/ATR DNA-damage response (Gasser et al., Nature 2005)
        is the mechanism, and a DNA-damage state variable on CellState
        would be the right driver. There isn't one yet, and the two
        available proxies are weak: S-phase occupancy is ~0 at demo
        timescales, and hypoxia saturates without the vascular module in
        the run. So mutation count is the honest stand-in and is named
        as such rather than dressed up as a stress-ligand field.
        """
        kwargs = dict(
            mhc1_expression=getattr(target_cell, 'mhc1_expression', 1.0),
            n_mutations=len(getattr(target_cell, 'mutations', None) or ()),
            max_kill_probability=self.nk_kill_probability,
            activation_threshold=self.nk_activation_threshold,
        )
        name = getattr(immune_cell, 'kir_subset', None)
        subset = (self._nk_licensing.subset_for(name)
                  if name is not None else None)
        if subset is not None:
            return nk_clone_decision(subset, **kwargs)
        return nk_decision(self._nk_licensing, **kwargs)

    def _nk_kill_probability(self, immune_cell, target_cell) -> float:
        """Per-encounter NK kill probability for this clone and target."""
        if self._nk_licensing is None:
            return self.nk_kill_probability
        return self._nk_decision(immune_cell, target_cell).kill_probability

    def _macrophage_kill_probability(
        self, target_cell, immune_cell: Optional[ImmuneCell] = None,
    ) -> float:
        """Per-encounter phagocytic probability: PD-1 gate x polarization.

        Two independent mechanisms compose multiplicatively, one reading
        the TARGET and one reading the EFFECTOR.

        Target side -- tumour PD-L1 engaging macrophage PD-1 suppresses
        phagocytosis, and checkpoint blockade relieves it (Gordon et
        al., Nature 2017, 545:495-499). ``pdl1_expression`` is exactly
        0.0 on a cell that neither IFN-gamma from an activated T cell
        nor an oncogenic driver has touched, and the gate is then the
        configured baseline to the bit.

        Effector side -- M1 macrophages are more tumoricidal than M2
        (Mills CD et al., J Immunol 2000; Mantovani A et al., Trends
        Immunol 2002; Qian BZ & Pollard JW, Cell 2010). ``kill_factor``
        is exactly 1.0 at M0, so an unpolarized macrophage keeps the
        baseline unchanged.

        Both factors are therefore exact identities at the untouched
        default state, which is what makes this a provable no-op on the
        existing baseline and a real effect in a PD-L1-high or
        M2-skewed TME. ``immune_cell`` is optional so a caller with only
        a target in hand still gets the target-side gate.
        """
        p = phagocytosis_probability(
            baseline=self.macrophage_kill_probability,
            pdl1_expression=getattr(target_cell, 'pdl1_expression', 0.0),
            checkpoint_block=self.checkpoint_block,
            max_suppression=self.macrophage_pd1_max_suppression,
        )
        if immune_cell is None:
            return p
        return min(1.0, p * macrophage_kill_factor(
            getattr(immune_cell, 'polarization', 0.0),
            span=self.macrophage_kill_span,
        ))

    def classify_tme(self, **kwargs) -> TMEClassification:
        """Classify the current tumor microenvironment into Teng's
        4-type scheme (Cancer Res 2015) and emit a TME_CLASSIFIED
        event with the result.

        Reads cancer cells from the linked cellular module and immune
        cells from this module's population. Keyword args are forwarded
        to ``engine.py.immune.tme_classifier.classify_tme`` so callers
        can override TIL / PD-L1 thresholds at the call site.

        Returns the TMEClassification dataclass; also emits a
        TME_CLASSIFIED event whose data dict contains the resolved
        ``tme_type`` (as a string), TIL / PD-L1 counts, and the
        clinical-readout fields ``predicted_icb_response`` and
        ``description``.
        """
        if self.cellular_module is None:
            raise RuntimeError(
                "ImmuneModule.classify_tme requires a linked cellular "
                "module; call set_cellular_module() first."
            )
        cancer_cells = [
            c for c in self.cellular_module.cells.values()
            if c.cell_type == 'cancer' and c.alive
        ]
        immune_cells = list(self.immune_cells.values())
        result = _classify_tme(cancer_cells, immune_cells, **kwargs)

        self.emit_event(EventTypes.TME_CLASSIFIED, {
            'tme_type': result.tme_type.value,
            'n_cancer_cells': result.n_cancer_cells,
            'n_til': result.n_til,
            'til_ratio': result.til_ratio,
            'pdl1_positive_fraction': result.pdl1_positive_fraction,
            'mean_ecm_density': result.mean_ecm_density,
            'ecm_excluded': result.ecm_excluded,
            'predicted_icb_response': result.predicted_icb_response,
            'description': result.description,
        })
        return result

    def update(self, dt: float):
        """Update immune system"""
        self._advance_clock(dt)
        if not self.cellular_module:
            return
        
        # Get cancer cells from cellular module
        cancer_cells = {cid: cell for cid, cell in self.cellular_module.cells.items()
                       if cell.cell_type == 'cancer' and cell.alive}
        
        for immune_id, immune_cell in list(self.immune_cells.items()):
            if immune_cell.in_blood:
                continue

            # M1/M2 polarization drifts toward its local cytokine set
            # point every step, whether or not the macrophage is
            # engaged. tau is 24 h, so at dt = 0.01 h one step closes
            # 1/2400 of the gap: the mechanism is real but slow, and a
            # 2 h demo run moves a macrophage only about 8% of the way
            # to its target. That is the timescale of macrophage
            # repolarization, not a defect -- if it needs to be visible
            # in a demo, the run duration is what should change.
            if (immune_cell.cell_type == 'macrophage'
                    and self.macrophage_polarization_enabled):
                self._update_polarization(immune_cell, dt)

            # Patrol (random walk), but only when there is nothing in reach.
            #
            # Detection is evaluated AFTER movement in this loop, so a cell
            # that steps away from a target it was already touching never
            # sees it. That is also wrong biologically: a lymphocyte arrests
            # on contact -- the synapse forms and migration stops -- it does
            # not wander off a cell it is sitting on and come back. Cells
            # were previously frozen by a unit bug, which hid this entirely.
            if not immune_cell.activated:
                if not self._target_within_reach(immune_cell, cancer_cells):
                    self._patrol(immune_cell, dt)

                # ECM barrier (Upgrade 6): the stromal density at the
                # T cell's position compresses its effective sensing
                # radius. In a fully fibrotic region a T cell may have
                # functionally zero detection range, even though
                # antigens are present nearby.
                local_ecm = self.cellular_module.ecm_density_at(
                    immune_cell.position
                )
                effective_detection = detection_attenuation(
                    immune_cell.detection_radius, local_ecm,
                )

                # Look for cancer cells
                for cancer_id, cancer_cell in cancer_cells.items():
                    distance = np.linalg.norm(immune_cell.position - cancer_cell.position)

                    if distance < effective_detection:
                        recognized, tcr_match = self._recognize(
                            immune_cell, cancer_cell
                        )
                        if recognized:
                            immune_cell.activated = True
                            immune_cell.target_cell_id = cancer_id
                            immune_cell.active_tcr_match = tcr_match
                            self.total_activations += 1
                            if tcr_match is not None:
                                self.total_tcr_recognitions += 1
                                # Adaptive PD-L1 induction (Type I TME
                                # mechanism): an activated TCR-pMHC
                                # engagement releases IFN-gamma, which
                                # paramedically up-regulates PD-L1 on
                                # the target tumor cell. Bumped per
                                # engagement, clamped to 1.0.
                                cancer_cell.pdl1_expression = float(min(
                                    1.0,
                                    cancer_cell.pdl1_expression + 0.5,
                                ))
                                # Exhaustion is counted per contact, in
                                # _engage: recognition happens at sensing
                                # range, and the TCR engages its pMHC only
                                # once the cells touch.

                            self.emit_event(EventTypes.IMMUNE_ACTIVATED, {
                                'immune_id': immune_id,
                                'immune_type': immune_cell.cell_type,
                                'target_id': cancer_id,
                                'position': immune_cell.position.tolist(),
                                'tcr_id': (
                                    tcr_match.tcr.tcr_id if tcr_match else None
                                ),
                                'affinity': (
                                    tcr_match.affinity if tcr_match else None
                                ),
                            })
                            break
            
            # Attack target
            else:
                if immune_cell.target_cell_id in cancer_cells:
                    target = cancer_cells[immune_cell.target_cell_id]

                    # Move toward target
                    direction = target.position - immune_cell.position
                    distance = np.linalg.norm(direction)

                    if distance > 0:
                        # ECM barrier (Upgrade 6) attenuates the T
                        # cell's speed at its current position. The
                        # stromal density on the path matters most;
                        # we sample at the T cell, which is the
                        # cheapest first-order approximation.
                        local_ecm = (
                            self.cellular_module.ecm_density_at(
                                immune_cell.position
                            )
                        )
                        effective_speed = motility_attenuation(
                            immune_cell.speed, local_ecm,
                        )
                        direction = direction / distance
                        immune_cell.velocity = direction * effective_speed
                        immune_cell.position += chase_displacement(
                            direction, distance, effective_speed, dt,
                        )
                    
                    # Kill if close enough. T cells with a TCR match use
                    # the closed-loop kill_probability(affinity, mhc,
                    # costim); NK uses the KIR-HLA receptor balance (or
                    # its own flat baseline with no typing on record)
                    # and macrophages the PD-1-gated phagocytic
                    # probability.
                    if distance < immune_cell.kill_radius:
                        self._engage(immune_cell, target, cancer_cells)
                else:
                    # Target lost, deactivate
                    self._release(immune_cell)
            
            # Keep in bounds
            immune_cell.position = np.clip(immune_cell.position, [20, 20, 20], [180, 180, 80])
    
    def _advance_clock(self, dt: float) -> None:
        """Advance the module clock contacts resolve against (hours)."""
        self.clock_h += dt

    def _engage(self, immune_cell: ImmuneCell, target, cancer_cells) -> None:
        """One step of an effector in contact with its target.

        A contact is an encounter with one outcome, decided when it begins.
        It kills with the per-contact probability _target_kill_probability
        gives -- the T-cell Hill rule, the NK receptor balance, the
        macrophage PD-1 gate -- and it lasts a time drawn from the lineage's
        measured contact duration: an exponential with the mean of its
        lethal contact (contact to the target's death) or its non-lethal one
        (contact to detachment). When the time is up, a lethal contact kills
        the target and a non-lethal one ends with the effector letting go,
        free to sense and engage again -- so killing that needs repeated
        contacts emerges from repeated tries, as imaging of CTLs in tumours
        shows it does (PMID:34471116).

        This replaced a roll of the same probability on every step of
        contact. That made the kill rate per hour a function of dt -- at up
        to 0.8 per roll a cell killed within a step or two of touching,
        whatever the step was -- and the T-cell exhaustion counter, which
        counted steps of contact against a threshold of five encounters,
        exhausted a clone after three minutes on one target at dt = 0.01 h.
        """
        now = self.clock_h
        if immune_cell.contact_ends_at is None:
            self._register_tcr_engagement(immune_cell)
            p_kill = self._target_kill_probability(immune_cell, target)
            immune_cell.contact_lethal = bool(np.random.random() < p_kill)
            minutes = self._contact_minutes_for(
                immune_cell.cell_type, immune_cell.contact_lethal)
            immune_cell.contact_ends_at = (
                now + np.random.exponential(minutes) / MINUTES_PER_HOUR)
            return
        if now < immune_cell.contact_ends_at:
            return
        if immune_cell.contact_lethal:
            # CANCER_KILLED is queued, so the cellular module removes the
            # target only when the step ends. Take it out of this step's
            # snapshot now, or a second cell reaching it in the same step
            # kills it again -- two kills, two events, one death -- and a
            # third detects it and activates on a corpse.
            cancer_cells.pop(immune_cell.target_cell_id, None)
            self._kill_target(immune_cell, target)
        else:
            self._release(immune_cell)

    def _release(self, immune_cell: ImmuneCell) -> None:
        """End an engagement: the effector disengages and patrols again."""
        immune_cell.activated = False
        immune_cell.target_cell_id = None
        immune_cell.active_tcr_match = None
        immune_cell.contact_ends_at = None
        immune_cell.contact_lethal = False

    def _register_tcr_engagement(self, immune_cell: ImmuneCell) -> None:
        """Count one TCR-pMHC engagement for a T cell's clone.

        Once per contact. Chronic antigen exposure is repeated contact
        with a target that does not die, which is what drives a clone from
        PD-1-lo precursor to PD-1-hi exhausted (Dolina et al. 2021).
        """
        if (immune_cell.cell_type != 'T_cell'
                or immune_cell.active_tcr_match is None):
            return
        match = immune_cell.active_tcr_match
        tcr_id = match.tcr.tcr_id
        new_count, did_exhaust = self._tcr_repertoire.register_engagement(tcr_id)
        if did_exhaust:
            self.total_exhaustion_transitions += 1
            pres = match.presentation
            self.emit_event(
                EventTypes.TCELL_EXHAUSTED,
                {
                    'tcr_id': tcr_id,
                    'cdr3': match.tcr.cdr3,
                    'encounter_count': new_count,
                    'target_id': immune_cell.target_cell_id,
                    'peptide': pres.peptide.sequence,
                    'mutation': pres.peptide.mutation_label,
                },
            )

    def _contact_minutes_for(self, lineage: str, lethal: bool) -> float:
        """Mean contact duration for ``lineage``, in minutes, read once."""
        field_name = CONTACT_FIELDS[0] if lethal else CONTACT_FIELDS[1]
        key = (lineage, field_name)
        minutes = self._contact_minutes.get(key)
        if minutes is not None:
            return minutes
        configured = (self.config.get(field_name) or {}).get(lineage)
        if configured is not None:
            minutes, note = float(configured), 'config'
        else:
            minutes, note = self._read_library_contact_minutes(lineage, field_name)
        self._contact_minutes[key] = minutes
        self.contact_time_source.setdefault(lineage, {})[field_name] = note
        return minutes

    #: Granules a cytotoxic cell sheds per target killed, and the floor
    #: it keeps. Measured on primary human NK cells: 2-4 degranulation
    #: events per target, about 10% of the reserve released per target
    #: and about 1% required (PMID 29592963). A killing cell sheds ~6 of
    #: 63 and stays loaded -- it does not empty itself, which is why one
    #: NK cell kills serially.
    GRANULES_PER_KILL = (2, 4)

    def _lytic_reserve(self, lineage: str) -> Optional[float]:
        """Granules this lineage carries, from the library, or None.

        None means the lineage has no sourced granule count -- which is
        the answer for a macrophage and for a helper T cell, and is not
        the same as zero.
        """
        if not self.library_params:
            return None
        try:
            from cognisom.genomics import parameter_resolver
            resolver = parameter_resolver.shared_resolver()
        except Exception as exc:                        # pragma: no cover
            logger.debug("entity library unavailable (%s)", exc)
            return None
        if resolver is None:
            return None
        name, entity_type = LINEAGE_ENTITY.get(lineage, (lineage, 'immune_cell'))
        pv, _note = resolver.sourced_value(name, entity_type,
                                           'lytic_granule_count')
        return float(pv.value) if pv is not None else None

    def _kill_route(self, immune_cell: "ImmuneCell") -> str:
        """Which of the two killing routes this cell used.

        A cytotoxic lymphocyte kills by granule exocytosis -- perforin
        opens the target membrane and granzyme B goes through it -- or
        through a death receptor, FasL on the killer engaging Fas on the
        target. They converge on caspase-3.

        The library holds both sets of molecules now. What it cannot
        decide is which route a given contact used, and nothing in this
        model distinguishes them: the kill is one Bernoulli draw. So this
        reports the route a cell is EQUIPPED for, from whether it carries
        a granule reserve, and says so rather than implying the model
        chose.
        """
        if getattr(immune_cell, 'lytic_granules', None):
            return 'granule_exocytosis'
        if immune_cell.cell_type in ('T_cell', 'NK_cell'):
            return 'death_receptor'
        return 'unresolved'

    def _read_library_contact_minutes(self, lineage: str, field_name: str):
        """(minutes, note): the library's value if it is evidence, else the default."""
        default = (DEFAULT_LETHAL_CONTACT_MIN if field_name == CONTACT_FIELDS[0]
                   else DEFAULT_NONLETHAL_CONTACT_MIN)
        if not self.library_params:
            return default, 'default (library_params=False)'
        try:
            from cognisom.genomics import parameter_resolver
            resolver = parameter_resolver.shared_resolver()
        except Exception as exc:                        # pragma: no cover
            logger.warning("entity library unavailable (%s)", exc)
            resolver = None
        if resolver is None:
            return default, 'default (library unavailable)'
        name, entity_type = LINEAGE_ENTITY.get(lineage, (lineage, 'immune_cell'))
        pv, note = resolver.sourced_value(name, entity_type, field_name)
        if pv is None:
            return default, f"default (illustrative); {name}: {note}"
        return float(pv.value), f"{name}: {note}"

    def _target_within_reach(self, immune_cell: ImmuneCell, cancer_cells) -> bool:
        """Is a cancer cell already inside this cell's sensing radius?

        Used to arrest migration before the random walk carries a cell off a
        target it is in contact with. Deliberately geometric and cheap: it
        asks only "is something there", not "do I recognise it" -- a cell
        pauses to inspect what it bumps into, and recognition is decided
        immediately afterwards by the detection pass.
        """
        reach = float(getattr(immune_cell, 'detection_radius', 0.0) or 0.0)
        if reach <= 0.0 or not cancer_cells:
            return False
        for cancer_cell in cancer_cells.values():
            if np.linalg.norm(
                    immune_cell.position - cancer_cell.position) < reach:
                return True
        return False

    def _patrol_speed_for(self, lineage: str) -> float:
        """Patrol speed of ``lineage`` in um/min, reading the library once.

        Called for every patrol step, so after the first call for a
        lineage it is dictionary lookups only. Cells built outside
        add_immune_cell (the ingestion bridge constructs ImmuneCell
        directly) are resolved here on their first step.
        """
        speed = self.lineage_patrol_speed.get(lineage)
        if speed is not None:
            return speed
        if lineage not in self.patrol_speed_source:
            self.patrol_speed_source[lineage] = (
                self._read_library_patrol_speed(lineage)
            )
            return self.lineage_patrol_speed.get(lineage, self.patrol_speed)
        return self.patrol_speed

    def _read_library_patrol_speed(self, lineage: str) -> str:
        """Take ``lineage``'s migration speed from the library if it is evidence.

        Returns the note recorded in patrol_speed_source. The resolver
        passes a value only at simulation grade (never a PLACEHOLDER --
        the cell_type Macrophage row's stamped 0.5 um/min would otherwise
        slow every macrophage tenfold) and only in the spec's unit, um/min,
        which is the unit patrol_speed is quoted in; the hour-step
        conversion happens once, in _patrol. Fails soft: a deploy without
        the library still constructs an engine.
        """
        if not self.library_params:
            return 'library not consulted (library_params=False)'
        try:
            from cognisom.genomics import parameter_resolver
            resolver = parameter_resolver.shared_resolver()
        except Exception as exc:                        # pragma: no cover
            logger.warning("entity library unavailable (%s)", exc)
            resolver = None
        if resolver is None:
            return 'library unavailable'
        name, entity_type = LINEAGE_ENTITY.get(lineage, (lineage, 'immune_cell'))
        pv, note = resolver.sourced_value(
            name, entity_type, 'migration_speed_um_per_min',
        )
        if pv is not None:
            self.lineage_patrol_speed[lineage] = float(pv.value)
            logger.info("%s patrols at %.3g um/min from the library (%s, %s)",
                        lineage, pv.value, name, note)
        return f"{name}: {note}"

    def _patrol(self, immune_cell: ImmuneCell, dt: float):
        """Random patrol movement, ECM-attenuated."""
        # Random walk, biased up the chemokine gradient where there is
        # one. A biased random walk IS chemotaxis as leukocytes do it:
        # the cell keeps its own motility and tilts it, which is why the
        # bias mixes directions and leaves the speed alone.
        direction = np.random.randn(3)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        field = self._chemotaxis_gradient_field()
        if field is not None:
            gradient = np.asarray(field.get_gradient_at(immune_cell.position),
                                  dtype=float)
            magnitude = float(np.linalg.norm(gradient))
            if magnitude > 0.0:
                w = self.chemotaxis_bias
                direction = (1.0 - w) * direction + w * (gradient / magnitude)
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                self.chemotaxis_steps += 1

        # ECM barrier (Upgrade 6): patrol speed scales down with the
        # local stromal density. A T cell wandering into a fibrotic
        # region effectively stalls.
        base_speed = self._patrol_speed_for(immune_cell.cell_type)
        effective_speed = base_speed
        if self.cellular_module is not None:
            local_ecm = self.cellular_module.ecm_density_at(
                immune_cell.position
            )
            effective_speed = motility_attenuation(base_speed, local_ecm)
        immune_cell.velocity = direction * effective_speed

        # patrol_speed is um/MIN and dt is HOURS, so the conversion is x60.
        # This read `* dt * 0.01`, which is 6,000x too slow: at the default
        # 5 um/min and dt=0.05 h a cell stepped 0.0025 um instead of 15 um,
        # and over a 6 h run the whole immune compartment moved a median of
        # 0.022 um -- frozen. Nothing ever reached a tumour: the nearest
        # NK-to-cancer gap stayed at 12.7 um against a 10 um detection
        # radius, so recognition (which works) was never even attempted.
        step = immune_cell.velocity * dt * MINUTES_PER_HOUR

        # Do not let a cell cross more than its own sensing radius in one
        # step. Detection is evaluated once per step, so a displacement
        # larger than detection_radius lets a cell step clean over a target
        # it should have found -- at dt=0.05 h the uncapped step is 15 um
        # against a 10 um radius, which measurably broke the closed-loop
        # trace (the T cell overshot the cancer cell it was meant to kill).
        # This is a discretisation guard, not a speed limit: shorten dt and
        # the cap stops binding.
        reach = float(getattr(immune_cell, 'detection_radius', 0.0) or 0.0)
        if reach > 0.0:
            length = float(np.linalg.norm(step))
            if length > reach:
                step = step * (reach / length)
        immune_cell.position += step
    
    def _recognize(self, immune_cell: ImmuneCell, cancer_cell):
        """Decide whether this immune cell recognizes this cancer cell.

        Returns (recognized: bool, tcr_match: Optional[TCRMatch]).
        TCRMatch is populated only for T-cell recognitions; NK and
        macrophage paths keep the legacy threshold heuristic.
        """
        if immune_cell.cell_type == 'T_cell':
            # Upgrade 2: TCR-pMHC matching. The cancer cell must have at
            # least one MHC-I-displayed neoantigen, AND a clone in the
            # patient TCR repertoire must clear the recognition
            # threshold against one of those presentations.
            presentations: List[MHCPresentation] = getattr(
                cancer_cell, 'mhc1_displayed_peptides', []
            )
            if not presentations or self._tcr_repertoire is None:
                return False, None
            # MHC-I downregulation gates display visibility too: a
            # cancer cell with mhc1_expression near 0 hides its
            # neoantigens. We require the displayed pool to be non-
            # empty *and* the surface expression to be above a small
            # epsilon -- the patent claim is the closed loop, not the
            # surface gate, so the threshold is intentionally lenient.
            if cancer_cell.mhc1_expression < 0.05:
                return False, None
            for pres in presentations:
                match = self._tcr_repertoire.best_match(pres)
                if match is not None and match.is_recognized:
                    return True, match
            return False, None

        elif immune_cell.cell_type == 'NK_cell':
            # Activating / inhibitory receptor balance. With the
            # patient's class-I typing on record, engagement is decided
            # by whether any subset of their NK compartment integrates a
            # net activating signal -- KIR-HLA licensing (Kim 2005,
            # Anfossi 2006) over missing self (Karre 1986). Without a
            # typing, the legacy step rule applies, but its threshold is
            # at least visible in config now instead of being a bare
            # 0.4 literal here. Measured, the old rule was a hard step:
            # identical kills at MHC-I 0.00 / 0.20 / 0.39 and none at
            # 0.40 / 0.41 / 0.80 / 1.00.
            if self._nk_licensing is not None:
                return self._nk_decision(
                    immune_cell, cancer_cell,
                ).recognized, None
            return missing_self_recognized(
                cancer_cell.mhc1_expression,
                self.nk_missing_self_threshold,
            ), None

        elif immune_cell.cell_type == 'macrophage':
            # Macrophage targeting is not antigen-restricted, so there
            # is no MHC gate to apply here -- but `return True` ignored
            # the don't-eat-me side entirely, and macrophage-only runs
            # measured identical kills at target MHC-I 0.0, 0.4 and 1.0.
            # Engagement now requires the PD-1 gate to leave some
            # phagocytic capacity; at PD-L1 = 0 that is the configured
            # baseline, so an untouched cell behaves exactly as before.
            # An M2 macrophage is pro-tumour, not tumoricidal, and
            # stops recognizing targets at all past the configured
            # polarization. The default cutoff is on the far side of
            # M0, so an unpolarized macrophage still recognizes
            # everything exactly as `return True` did.
            if (getattr(immune_cell, 'polarization', 0.0)
                    > self.macrophage_recognition_max_polarization):
                return False, None
            return self._macrophage_kill_probability(
                cancer_cell, immune_cell,
            ) > 0.0, None

        return False, None

    def _live_is_exhausted(self, match) -> bool:
        """Resolve a TCR clone's exhaustion state from the repertoire.

        ``TCRMatch`` is a *snapshot* taken at recognition time and cached
        on the immune cell as ``active_tcr_match``. It is rewritten only
        when the T cell re-recognizes a target, and cleared on target
        loss or kill. But exhaustion accrues on the clone, once per
        contact by any of its cells (``_register_tcr_engagement``), so the
        clone can cross the threshold while this cell's snapshot still
        reports PRECURSOR.

        Reading ``match.is_exhausted`` would miss that transition, which
        is precisely the chronic-antigen-exposure regime the two-state
        model exists to represent. Resolving against the repertoire by
        clone id keeps the ICB-rescue gate honest.

        Falls back to the snapshot when the clone is unknown to the
        repertoire (repertoire replaced mid-run, or a match constructed
        by a caller outside the normal recognition path).
        """
        if self._tcr_repertoire is None:
            return bool(match.is_exhausted)
        tcr = getattr(match, 'tcr', None)
        tcr_id = getattr(tcr, 'tcr_id', None)
        if tcr_id is None:
            return bool(match.is_exhausted)
        try:
            return (
                self._tcr_repertoire.exhaustion_state(tcr_id)
                is ExhaustionState.EXHAUSTED
            )
        except KeyError:
            return bool(match.is_exhausted)

    def _target_kill_probability(self, immune_cell: ImmuneCell, target_cell) -> float:
        """Per-encounter kill probability, per lineage.

        T cells with an active TCR match use the affinity * MHC * costim
        Hill rule from tcell_kill.kill_probability, then the sympathetic
        gate. NK cells use the KIR-HLA activating/inhibitory balance
        when the patient's class-I typing is on record and their own
        flat baseline otherwise. Macrophages use the PD-1-gated
        phagocytic probability. The three used to be one number.

        The final `return self.kill_probability` is now reachable only
        for an immune cell of an unrecognized type; it is not a T-cell
        fallback. See the comment on self.kill_probability in __init__.
        """
        if (immune_cell.cell_type == 'T_cell'
                and immune_cell.active_tcr_match is not None):
            match = immune_cell.active_tcr_match
            outcome = kill_outcome(
                affinity=match.affinity,
                mhc_level=getattr(target_cell, 'mhc1_expression', 1.0),
                costimulation=self.costimulation,
                checkpoint_block=self.checkpoint_block,
                # Lecture slide 52: exhausted clones cannot be rescued
                # by checkpoint blockade. The exhaustion state is read
                # LIVE from the repertoire rather than from the cached
                # TCRMatch -- see _live_is_exhausted for why.
                is_exhausted=self._live_is_exhausted(match),
            )
            # Sympathetic suppression (Upgrade 7): β2AR drive scales
            # T-cell function down; β-blocker rescues. Composes with
            # the affinity / MHC / costim / exhaustion axes as a
            # final multiplicative gate -- the cleanest place to add
            # it because every other knob has already resolved.
            sympathetic_mult = sympathetic_attenuation(
                self.stress_level,
                self.beta_blocker,
                max_suppression=self.sympathetic_max_suppression,
            )
            # Arginase-mediated M2 suppression: local M2 macrophages
            # express arginase-1 and deplete extracellular L-arginine,
            # downregulating CD3-zeta and blocking T-cell function
            # (Rodriguez PC et al., Cancer Res 2004;64:5839-49; Bronte V
            # & Zanovello P, Nat Rev Immunol 2005;5:641-54). A third
            # multiplicative gate alongside exhaustion and the
            # sympathetic axis. Exactly 1.0 when no M2 macrophage is
            # within the sampling radius, so a configuration with no
            # macrophages -- which is every immune configuration in the
            # test suite, including the flagship closed loop -- is a
            # provable no-op on the T-cell path.
            arginase_mult = arginase_attenuation(
                self._local_m2_burden(immune_cell.position),
                max_suppression=self.arginase_max_suppression,
            )
            lactate_mult = self._lactate_attenuation_at(immune_cell)
            return (
                outcome.kill_probability * sympathetic_mult * arginase_mult * lactate_mult
            )

        if immune_cell.cell_type == 'NK_cell':
            return (self._nk_kill_probability(immune_cell, target_cell)
                    * self._lactate_attenuation_at(immune_cell))
        if immune_cell.cell_type == 'macrophage':
            return self._macrophage_kill_probability(
                target_cell, immune_cell,
            )
        return self.kill_probability

    def _lactate_attenuation_at(self, immune_cell: ImmuneCell) -> float:
        """Kill function retained at the effector's interstitial lactate;
        exactly 1.0 without a lactate field or at normal tissue levels."""
        fields = getattr(self.spatial_module, 'fields', None) or {}
        field = fields.get('lactate')
        if field is None or self.lactate_max_suppression <= 0.0:
            return 1.0
        return lactate_attenuation(field.get_concentration_at(immune_cell.position),
                                   max_suppression=self.lactate_max_suppression)

    def _kill_target(self, immune_cell: ImmuneCell, target_cell):
        """Kill target cancer cell.

        Always emits CANCER_KILLED (backwards-compat with everything
        subscribed to it). T-cell kills additionally emit
        CELL_KILLED_BY_TCELL so the closed-loop event trace
        MUTATION_OCCURRED -> PEPTIDE_GENERATED -> PEPTIDE_PRESENTED ->
        CELL_KILLED_BY_TCELL is recoverable from the event log.
        """
        # What the kill was done WITH. The library holds the effectors
        # now -- perforin, the granzymes, FasL, Fas, TRAIL, wired to the
        # caspases that were already there -- and until this, nothing in
        # the simulation referred to any of them: a kill was a counter.
        route = self._kill_route(immune_cell)
        released = 0
        reserve = getattr(immune_cell, 'lytic_granules', None)
        if route == 'granule_exocytosis' and reserve:
            low, high = self.GRANULES_PER_KILL
            # Its OWN stream. Drawing from the global numpy RNG would
            # consume numbers the rest of the simulation was going to
            # use, so every draw after the first kill shifts and the
            # whole trajectory moves -- measured here as a pinned
            # engagement count going from 6 to 2 and a hypoxic death
            # count from 60 to 58, with nothing about hypoxia changed.
            # An addition that is meant to record what happened must not
            # alter what happens.
            released = int(self._granule_rng.integers(low, high + 1))
            released = min(released, int(reserve))
            immune_cell.lytic_granules = max(0.0, reserve - released)

        self.emit_event(EventTypes.CANCER_KILLED, {
            'cell_id': target_cell.cell_id,
            'killer_id': immune_cell.cell_id,
            'killer_type': immune_cell.cell_type,
            'position': target_cell.position.tolist(),
            'route': route,
            'granules_released': released,
            'granules_remaining': getattr(immune_cell, 'lytic_granules', None),
        })
        self.total_kills += 1
        self.total_granules_released += released

        if (immune_cell.cell_type == 'T_cell'
                and immune_cell.active_tcr_match is not None):
            match = immune_cell.active_tcr_match
            self.total_tcell_kills += 1
            self.emit_event(EventTypes.CELL_KILLED_BY_TCELL, {
                'cell_id': target_cell.cell_id,
                'killer_id': immune_cell.cell_id,
                'tcr_id': match.tcr.tcr_id,
                'peptide': match.presentation.peptide.sequence,
                'hla_allele': match.presentation.hla_allele,
                'mutation': match.presentation.peptide.mutation_label,
                'source_gene': match.presentation.peptide.source_gene,
                'affinity': match.affinity,
                'position': target_cell.position.tolist(),
            })

        # Deactivate immune cell
        self._release(immune_cell)

    # ---- Macrophage M1/M2 polarization -------------------------

    def _immune_neighbours(self, position, exclude_id=None):
        """Immune cells in tissue within the cytokine radius.

        On-demand neighbour sampling, following the precedent of
        spatial/ecm_barrier.ecm_density_at. O(n) per call, which is what
        the recognition loop around it already costs.
        """
        out = []
        r2 = self.cytokine_sampling_radius * self.cytokine_sampling_radius
        for cell in self.immune_cells.values():
            if cell.in_blood or cell.cell_id == exclude_id:
                continue
            d = cell.position - position
            if float(np.dot(d, d)) <= r2:
                out.append(cell)
        return out

    def _local_ifn_gamma(self, position, exclude_id=None,
                         neighbours=None) -> float:
        """Local IFN-gamma from activated T cells, in [0, 1].

        Activated CD8 T cells are the IFN-gamma source. There is no
        cytokine field and no diffusion solver here, and none is
        claimed: the signal is counted from nearby cells on demand. The
        30 um sampling radius is the measured effective range of
        paracrine IFN-gamma / IL-2 signalling (Oyler-Yaniv A et al.,
        Immunity 2017;46:609-20) rather than an arbitrary number.
        ``neighbours`` lets a caller that already has the neighbour list
        reuse it instead of rescanning.
        """
        if neighbours is None:
            neighbours = self._immune_neighbours(position, exclude_id)
        n = sum(
            1 for c in neighbours
            if c.cell_type == 'T_cell' and c.activated
        )
        return local_signal(
            n, saturation_count=self.cytokine_saturation_count)

    def _local_m2_secretion(self, position, species, exclude_id=None,
                            neighbours=None) -> float:
        """Local level of one M2 species from nearby macrophages.

        Each neighbour contributes its own ``cytokine_output``, so the
        signal is graded by how M2 each macrophage actually is rather
        than by a label. Exactly 0.0 with no macrophage nearby.
        """
        if neighbours is None:
            neighbours = self._immune_neighbours(position, exclude_id)
        total = 0.0
        for c in neighbours:
            if c.cell_type != 'macrophage':
                continue
            total += cytokine_output(
                getattr(c, 'polarization', 0.0))[species]
        return local_signal(
            total, saturation_count=self.cytokine_saturation_count)

    def _local_m2_burden(self, position, exclude_id=None) -> float:
        """Local arginase-1 burden from M2 macrophages, in [0, 1].

        Exactly 0.0 when no macrophage is nearby, which is what makes
        the arginase gate on the T-cell path a provable no-op in any
        configuration with no macrophages -- including the flagship
        closed loop, which configures none.
        """
        return self._local_m2_secretion(position, 'ARG1', exclude_id)

    def _local_hypoxia(self, position) -> float:
        """Hypoxia drive at a position, from the nearest living cell.

        Returns 0.0 with no cellular module linked and 0.0 when no cell
        is in range: absent oxygen data is not hypoxia. Hypoxia and
        tumour-derived lactate push tumour-associated macrophages toward
        M2 (Colegio OR et al., Nature 2014;513:559-63; Henze AT &
        Mazzone M, J Clin Invest 2016;126:3672-9). Note this needs a
        VascularModule in the run to read a graded field -- without one
        the tissue dies off and oxygen stops being informative.
        """
        if self.cellular_module is None:
            return 0.0
        best = None
        best_d2 = self.cytokine_sampling_radius ** 2
        for cell in self.cellular_module.cells.values():
            if not cell.alive:
                continue
            d = cell.position - position
            d2 = float(np.dot(d, d))
            if d2 <= best_d2:
                best_d2 = d2
                best = cell
        if best is None:
            return 0.0
        return hypoxia_signal(getattr(best, 'oxygen', NORMOXIC_OXYGEN))

    def _update_polarization(self, immune_cell: ImmuneCell, dt: float):
        """One polarization relaxation step for one macrophage."""
        pos = immune_cell.position
        own = immune_cell.cell_id
        # One neighbour scan feeds both cytokine arms. The recognition
        # loop around this already pays two O(n) ecm_density_at scans
        # per immune cell per step, so the added cost is bounded by that
        # precedent rather than new in kind -- but there is no reason to
        # pay it three times over. Measured: merging the scans cut the
        # overhead at 8 macrophages from 1.34x to 1.20x, with
        # byte-identical polarization values.
        neighbours = self._immune_neighbours(pos, own)
        m2_signal = self._local_m2_secretion(
            pos, 'IL10', own, neighbours)
        target = polarization_target(
            ifn_gamma=self._local_ifn_gamma(pos, own, neighbours),
            # One IL-10 / TGF-beta term, not two: the simulated M2
            # source secretes both and they act on the same arm, so
            # splitting them would double-count one signal.
            il10=m2_signal,
            tgf_beta=0.0,
            hypoxia=self._local_hypoxia(pos),
        )
        immune_cell.polarization = polarization_step(
            immune_cell.polarization,
            target,
            dt,
            tau_hours=self.polarization_tau_hours,
        )

    def get_innate_state(self) -> Dict[str, Any]:
        """Innate-arm readout: NK licensing, macrophage polarization, and
        the provenance of each.

        Both halves report provenance separately because they are
        measured to different degrees. The NK ligand side is real
        patient data (OptiType class-I typing from read data); the KIR
        side is a population carrier-frequency prior, and
        ``nk_kir_patient_specific`` says so. The macrophage seed is
        'unmeasured' unless a caller both supplied an M2 fraction and
        declared it measured -- and no patient in this platform has a
        measured M1/M2 ratio today, so 'unmeasured' is the correct
        answer and the default.

        """
        macrophages = [
            c for c in self.immune_cells.values()
            if c.cell_type == 'macrophage' and not c.in_blood
        ]
        pols = [c.polarization for c in macrophages]
        labels: Dict[str, int] = {}
        for c in macrophages:
            label = polarization_label(c.polarization)
            labels[label] = labels.get(label, 0) + 1

        if self.macrophage_m2_fraction is None:
            source = 'unmeasured'
        elif self.immune_profile_measured:
            source = 'patient_measured'
        else:
            source = 'configured_unmeasured'

        lic = self._nk_licensing
        return {
            # NK arm
            'nk_model': (
                'kir_hla_balance' if lic is not None
                else 'missing_self_threshold'
            ),
            'nk_kill_probability': self.nk_kill_probability,
            'nk_missing_self_threshold': self.nk_missing_self_threshold,
            'nk_patient_specific': (
                lic.hla.is_patient_specific if lic is not None else False
            ),
            # The KIR half is a population carrier-frequency prior, not
            # a measurement: nothing in this platform types KIR.
            'nk_kir_patient_specific': (
                lic.kir.is_patient_specific if lic is not None else False
            ),
            'nk_licensed_fraction': (
                lic.licensed_fraction if lic is not None else None
            ),
            'nk_ligand_groups': (
                [g.value for g in lic.hla.groups] if lic is not None else None
            ),
            # Macrophage arm
            'macrophage_kill_probability': self.macrophage_kill_probability,
            'n_macrophages_in_tissue': len(macrophages),
            'mean_macrophage_polarization': (
                float(sum(pols) / len(pols)) if pols else None
            ),
            'macrophage_polarization_counts': labels,
            'macrophage_polarization_enabled':
                self.macrophage_polarization_enabled,
            'macrophage_m2_fraction': self.macrophage_m2_fraction,
            'macrophage_polarization_source': source,
            'macrophage_pd1_max_suppression':
                self.macrophage_pd1_max_suppression,
            'immune_profile_measured': self.immune_profile_measured,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return current immune state"""
        active_immune = [ic for ic in self.immune_cells.values() if not ic.in_blood]
        activated_immune = [ic for ic in active_immune if ic.activated]
        
        t_cells = [ic for ic in active_immune if ic.cell_type == 'T_cell']
        nk_cells = [ic for ic in active_immune if ic.cell_type == 'NK_cell']
        macrophages = [ic for ic in active_immune if ic.cell_type == 'macrophage']
        
        return {
            'n_immune_cells': len(active_immune),
            'n_activated': len(activated_immune),
            'n_t_cells': len(t_cells),
            'n_nk_cells': len(nk_cells),
            'n_macrophages': len(macrophages),
            'total_kills': self.total_kills,
            'total_granules_released': self.total_granules_released,
            'lytic_reserves': {
                c.cell_id: getattr(c, 'lytic_granules', None)
                for c in self.immune_cells.values()
                if getattr(c, 'lytic_granules', None) is not None
            },
            'total_activations': self.total_activations,
            'total_recruited': self.total_recruited,
            # Innate arm. nk_model names the rule that actually ran.
            # The HLA and KIR halves of patient-specificity are reported
            # separately because only the HLA half can be measured here:
            # the KIR half is a population carrier-frequency prior.
            'nk_model': (
                'kir_hla_balance' if self._nk_licensing is not None
                else 'missing_self_threshold'
            ),
            'nk_patient_specific': (
                self._nk_licensing.hla.is_patient_specific
                if self._nk_licensing is not None else False
            ),
            'nk_kir_patient_specific': (
                self._nk_licensing.kir.is_patient_specific
                if self._nk_licensing is not None else False
            ),
            'nk_licensed_fraction': (
                self._nk_licensing.licensed_fraction
                if self._nk_licensing is not None else None
            ),
            'nk_ligand_groups': (
                [g.value for g in self._nk_licensing.hla.groups]
                if self._nk_licensing is not None else None
            ),
            # Macrophage arm. None when there are no macrophages in
            # tissue -- absent, not zero, the shape twin_config now
            # uses. The full readout with provenance is
            # get_macrophage_state().
            'mean_macrophage_polarization': (
                float(sum(getattr(ic, 'polarization', 0.0)
                          for ic in macrophages)
                      / len(macrophages)) if macrophages else None
            ),
            'macrophage_polarization_source': (
                'unmeasured' if self.macrophage_m2_fraction is None
                else ('patient_measured' if self.immune_profile_measured
                      else 'configured_unmeasured')
            ),
            # The legacy step rule's threshold, which used to be a bare
            # 0.4 literal inside _recognize and is now visible.
            'nk_missing_self_threshold': self.nk_missing_self_threshold,
            # Where each lineage's patrol speed came from: config, a cited
            # library value, or the reason the library had none to give.
            'patrol_speed_source': dict(self.patrol_speed_source),
            'contact_time_source': {k: dict(v) for k, v in
                                    self.contact_time_source.items()},
        }
    
    # Event handlers
    def on_cell_transformed(self, data):
        """Handle cell transformation - recruit more immune cells"""
        # Recruit additional immune cell
        position = data['position']
        
        # Add new T cell near transformation site
        offset = np.random.randn(3) * 20
        new_pos = np.array(position) + offset
        
        self.add_immune_cell(new_pos, 'T_cell')
        self.total_recruited += 1
        
        self.emit_event(EventTypes.IMMUNE_RECRUITED, {
            'cell_type': 'T_cell',
            'position': new_pos.tolist(),
            'reason': 'cell_transformation'
        })
    
    def on_cell_divided(self, data):
        """Handle cell division - track population"""
        # Could recruit more immune cells if cancer is dividing rapidly
        if data['cell_type'] == 'cancer':
            if np.random.random() < 0.1:  # 10% chance
                position = data['position']
                offset = np.random.randn(3) * 20
                new_pos = np.array(position) + offset
                
                self.add_immune_cell(new_pos, 'NK_cell')
                self.total_recruited += 1


# Test
if __name__ == '__main__':
    from cognisom.core import SimulationEngine, SimulationConfig
    from cellular_module import CellularModule
    
    print("=" * 70)
    print("Immune Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
    
    # Register modules
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 10,
        'n_cancer_cells': 5,
        'division_time_cancer': 10.0  # Slow for testing
    })
    
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 5,
        'n_nk_cells': 3,
        'n_macrophages': 2
    })
    
    # Initialize
    engine.initialize()
    
    # Link modules
    immune = engine.modules['immune']
    cellular = engine.modules['cellular']
    immune.set_cellular_module(cellular)
    
    print("Modules linked")
    print()
    
    # Run simulation
    engine.run()
    
    # Get results
    print("\nImmune State:")
    immune_state = immune.get_state()
    for key, value in immune_state.items():
        print(f"  {key}: {value}")
    
    print("\nCellular State:")
    cellular_state = cellular.get_state()
    for key, value in cellular_state.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("✓ Immune module working!")
    print("=" * 70)
