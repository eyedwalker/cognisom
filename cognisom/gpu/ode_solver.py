"""
Batched ODE Solver
==================

Solves the same ODE system across many cells at once, each cell with its
own parameters.

What is implemented, precisely:

- ``method='rk45'`` — adaptive Dormand-Prince 5(4) with an embedded error
  estimate and a PI step-size controller. Honours ``rtol``/``atol``, and
  rejects and retries a step that misses them. Explicit, so it is the
  right choice for non-stiff systems.
- ``method='bdf'`` — backward Euler (BDF of order 1) with a fixed three
  Newton iterations and a fixed step size. Suitable for stiff systems,
  but it is *not* variable order and it does not adapt its step, so
  ``rtol``/``atol`` do not constrain it.
- ``method='adams'`` — Adams-Moulton is not implemented. Requesting it
  logs a warning and runs the adaptive RK45 above.

Batching note: all cells share one time step, and the RK45 error norm is
taken over the whole batch. Cells advance together; there is no per-cell
step size.

GPU note: the CUDA path covers the BDF Newton iteration for the
2-species gene-expression model only. Every other system, and all of
RK45, runs in NumPy — the explicit stages call a NumPy ``rhs_func``, so
a device round-trip per stage would cost more than it saved.

VCell Parity Phase 1 - Foundation for hybrid ODE/SSA methods.

Usage::

    from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

    # Define a simple gene expression model
    system = ODESystem.gene_expression_cascade()

    solver = BatchedODEIntegrator(system, n_cells=10000, method='bdf')
    solver.integrate(t_span=(0, 10), y0=initial_conditions)

    result = solver.get_state()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .backend import get_backend
from .physics_interface import (
    BasePhysicsModel,
    PhysicsBackendType,
    PhysicsModelType,
    PhysicsState,
    register_physics,
)

log = logging.getLogger(__name__)
logger = log  # alias: both names are used in this module

# ── CUDA Kernels for BDF Integration ─────────────────────────────────────

_BDF_RHS_KERNEL = r"""
extern "C" __global__
void bdf_rhs_evaluation(
    const float* y,          // (n_cells, n_species) current state
    float* f,                // (n_cells, n_species) output derivatives
    const float* params,     // (n_params,) shared parameters
    const float* cell_params,// (n_cells, n_cell_params) per-cell parameters
    int n_cells,
    int n_species,
    int n_params,
    int n_cell_params
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    int base = cell * n_species;

    // Generic mass-action kinetics RHS
    // Each species has production/degradation terms
    // This is a template - actual RHS is problem-specific

    // Example: simple gene expression
    // dy0/dt = k_prod - k_deg * y0  (mRNA)
    // dy1/dt = k_trans * y0 - k_deg_prot * y1  (protein)

    // For now, use per-cell params: [k_prod, k_deg, k_trans, k_deg_prot]
    float k_prod = cell_params[cell * n_cell_params + 0];
    float k_deg = cell_params[cell * n_cell_params + 1];
    float k_trans = cell_params[cell * n_cell_params + 2];
    float k_deg_prot = cell_params[cell * n_cell_params + 3];

    float mRNA = y[base + 0];
    float protein = y[base + 1];

    f[base + 0] = k_prod - k_deg * mRNA;
    f[base + 1] = k_trans * mRNA - k_deg_prot * protein;
}
"""

_BDF_JACOBIAN_KERNEL = r"""
extern "C" __global__
void bdf_jacobian_eval(
    const float* y,          // (n_cells, n_species)
    float* J,                // (n_cells, n_species, n_species) Jacobian
    const float* cell_params,// (n_cells, n_cell_params)
    int n_cells,
    int n_species,
    int n_cell_params
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    int j_base = cell * n_species * n_species;

    // Zero out Jacobian
    for (int i = 0; i < n_species * n_species; i++) {
        J[j_base + i] = 0.0f;
    }

    // Fill in Jacobian entries (df_i / dy_j)
    float k_deg = cell_params[cell * n_cell_params + 1];
    float k_trans = cell_params[cell * n_cell_params + 2];
    float k_deg_prot = cell_params[cell * n_cell_params + 3];

    // df0/dy0 = -k_deg, df0/dy1 = 0
    // df1/dy0 = k_trans, df1/dy1 = -k_deg_prot
    J[j_base + 0*n_species + 0] = -k_deg;
    J[j_base + 0*n_species + 1] = 0.0f;
    J[j_base + 1*n_species + 0] = k_trans;
    J[j_base + 1*n_species + 1] = -k_deg_prot;
}
"""

_BDF_NEWTON_KERNEL = r"""
extern "C" __global__
void bdf_newton_step(
    float* y_new,            // (n_cells, n_species) updated solution
    const float* y_pred,     // (n_cells, n_species) predictor
    const float* f,          // (n_cells, n_species) RHS at y_new
    const float* J,          // (n_cells, n_species, n_species) Jacobian
    float dt,
    float gamma,             // BDF coefficient
    int n_cells,
    int n_species
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    int base = cell * n_species;
    int j_base = cell * n_species * n_species;

    // Residual: r = y_new - y_pred - gamma * dt * f(y_new)
    // Newton: (I - gamma*dt*J) * delta = -r
    // For 2x2 system, solve directly

    if (n_species == 2) {
        float a11 = 1.0f - gamma * dt * J[j_base + 0];
        float a12 = -gamma * dt * J[j_base + 1];
        float a21 = -gamma * dt * J[j_base + 2];
        float a22 = 1.0f - gamma * dt * J[j_base + 3];

        float r0 = y_new[base + 0] - y_pred[base + 0] - gamma * dt * f[base + 0];
        float r1 = y_new[base + 1] - y_pred[base + 1] - gamma * dt * f[base + 1];

        float det = a11 * a22 - a12 * a21;
        if (fabsf(det) > 1e-15f) {
            float delta0 = (a22 * (-r0) - a12 * (-r1)) / det;
            float delta1 = (a11 * (-r1) - a21 * (-r0)) / det;

            y_new[base + 0] += delta0;
            y_new[base + 1] += delta1;
        }
    }
    // For larger systems, would use LU decomposition
}
"""

_ERROR_ESTIMATION_KERNEL = r"""
extern "C" __global__
void estimate_error(
    const float* y_high,     // (n_cells, n_species) high-order solution
    const float* y_low,      // (n_cells, n_species) low-order solution
    float* error,            // (n_cells,) output error estimate
    float rtol,
    float atol,
    int n_cells,
    int n_species
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    int base = cell * n_species;
    float max_err = 0.0f;

    for (int i = 0; i < n_species; i++) {
        float y_h = y_high[base + i];
        float y_l = y_low[base + i];
        float scale = atol + rtol * fabsf(y_h);
        float err = fabsf(y_h - y_l) / scale;
        if (err > max_err) max_err = err;
    }

    error[cell] = max_err;
}
"""


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class ODESystem:
    """
    Definition of an ODE system for batched solving.

    Attributes
    ----------
    n_species : int
        Number of state variables per cell
    species_names : List[str]
        Names of species
    rhs_func : Optional[Callable]
        Python RHS function for CPU fallback: f(t, y, params) -> dydt
    rhs_kernel : Optional[str]
        CUDA kernel source for GPU RHS evaluation
    jacobian_func : Optional[Callable]
        Python Jacobian function for CPU (analytical or None for FD)
    parameters : Dict[str, float]
        Shared model parameters
    stiff : bool
        Whether the system is stiff (affects default method)
    """
    n_species: int
    species_names: List[str]
    rhs_func: Optional[Callable] = None
    rhs_kernel: Optional[str] = None
    jacobian_func: Optional[Callable] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    stiff: bool = True

    @staticmethod
    def gene_expression_2species() -> "ODESystem":
        """
        Simple 2-species gene expression model.

        Species:
            0: mRNA
            1: Protein

        Reactions:
            mRNA production:    ∅ → mRNA     (k_prod)
            mRNA degradation:   mRNA → ∅     (k_deg)
            Translation:        mRNA → mRNA + Protein  (k_trans)
            Protein degradation: Protein → ∅  (k_deg_prot)

        ODEs:
            d[mRNA]/dt = k_prod - k_deg * [mRNA]
            d[Protein]/dt = k_trans * [mRNA] - k_deg_prot * [Protein]
        """
        def rhs(t: float, y: np.ndarray, params: Dict) -> np.ndarray:
            mRNA, protein = y[..., 0], y[..., 1]
            dmRNA = params['k_prod'] - params['k_deg'] * mRNA
            dprotein = params['k_trans'] * mRNA - params['k_deg_prot'] * protein
            return np.stack([dmRNA, dprotein], axis=-1)

        def jacobian(t: float, y: np.ndarray, params: Dict) -> np.ndarray:
            n = y.shape[0] if y.ndim > 1 else 1
            J = np.zeros((n, 2, 2))
            J[..., 0, 0] = -params['k_deg']
            J[..., 0, 1] = 0.0
            J[..., 1, 0] = params['k_trans']
            J[..., 1, 1] = -params['k_deg_prot']
            return J

        return ODESystem(
            n_species=2,
            species_names=['mRNA', 'Protein'],
            rhs_func=rhs,
            jacobian_func=jacobian,
            parameters={
                'k_prod': 10.0,      # mRNA/hour
                'k_deg': 1.0,        # 1/hour (half-life ~42 min)
                'k_trans': 50.0,     # proteins/mRNA/hour
                'k_deg_prot': 0.5,   # 1/hour (half-life ~1.4 h)
            },
            stiff=False,
        )

    @staticmethod
    def ar_signaling_pathway() -> "ODESystem":
        """
        Androgen receptor signaling pathway (prostate-specific).

        Species:
            0: AR_mRNA
            1: AR_protein
            2: DHT (dihydrotestosterone)
            3: AR_DHT (active complex)
            4: PSA_mRNA
            5: PSA_protein
        """
        def rhs(t: float, y: np.ndarray, params: Dict) -> np.ndarray:
            AR_m, AR, DHT, AR_DHT, PSA_m, PSA = (
                y[..., 0], y[..., 1], y[..., 2],
                y[..., 3], y[..., 4], y[..., 5]
            )

            # AR dynamics
            dAR_m = params['k_ar_tx'] - params['k_ar_m_deg'] * AR_m
            dAR = params['k_ar_tl'] * AR_m - params['k_ar_deg'] * AR \
                  - params['k_bind'] * AR * DHT + params['k_unbind'] * AR_DHT

            # DHT dynamics (assumed constant input)
            dDHT = params['DHT_input'] - params['k_bind'] * AR * DHT \
                   + params['k_unbind'] * AR_DHT - params['k_dht_deg'] * DHT

            # AR-DHT complex
            dAR_DHT = params['k_bind'] * AR * DHT - params['k_unbind'] * AR_DHT \
                      - params['k_complex_deg'] * AR_DHT

            # PSA dynamics (induced by AR-DHT)
            dPSA_m = params['k_psa_tx'] * AR_DHT / (params['K_psa'] + AR_DHT) \
                     - params['k_psa_m_deg'] * PSA_m
            dPSA = params['k_psa_tl'] * PSA_m - params['k_psa_deg'] * PSA

            return np.stack([dAR_m, dAR, dDHT, dAR_DHT, dPSA_m, dPSA], axis=-1)

        return ODESystem(
            n_species=6,
            species_names=['AR_mRNA', 'AR', 'DHT', 'AR_DHT', 'PSA_mRNA', 'PSA'],
            rhs_func=rhs,
            parameters={
                'k_ar_tx': 5.0,        # AR transcription
                'k_ar_m_deg': 0.5,     # AR mRNA degradation
                'k_ar_tl': 20.0,       # AR translation
                'k_ar_deg': 0.1,       # AR protein degradation
                'k_bind': 100.0,       # AR-DHT binding
                'k_unbind': 10.0,      # AR-DHT unbinding
                'DHT_input': 1.0,      # DHT input rate
                'k_dht_deg': 0.2,      # DHT degradation
                'k_complex_deg': 0.05, # AR-DHT complex degradation
                'k_psa_tx': 2.0,       # PSA basal transcription
                'K_psa': 10.0,         # PSA induction K_m
                'k_psa_m_deg': 1.0,    # PSA mRNA degradation
                'k_psa_tl': 30.0,      # PSA translation
                'k_psa_deg': 0.2,      # PSA protein degradation
            },
            stiff=True,
        )


# ── Dormand-Prince 5(4) tableau ──────────────────────────────────────────
#
# Dormand & Prince, "A family of embedded Runge-Kutta formulae",
# J Comput Appl Math 6:19-26 (1980). The pair shares stages between a
# 5th-order and a 4th-order solution, so their difference is an error
# estimate that costs nothing extra -- which is what makes step-size
# control possible. The previous "rk45" was classical RK4: a single
# solution, no embedded pair, and therefore no way to measure error.
_DP_C = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])

_DP_A = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84],
]

# 5th-order weights (identical to the last A row: the method is FSAL).
_DP_B5 = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0])

# Difference between the 5th- and 4th-order weights, i.e. the error
# estimator applied directly to the stage derivatives.
_DP_E = np.array([
    35/384 - 5179/57600,
    0.0,
    500/1113 - 7571/16695,
    125/192 - 393/640,
    -2187/6784 + 92097/339200,
    11/84 - 187/2100,
    -1/40,
])

# PI controller exponents. Standard Hairer & Wanner values for an order-5
# pair: proportional term err^(-0.7/5), integral term err_prev^(0.4/5).
# A pure I-controller (err^(-1/5) alone) oscillates on mildly stiff
# problems; the PI form damps that.
_PI_ALPHA = 0.7 / 5.0
_PI_BETA = 0.4 / 5.0
_STEP_SAFETY = 0.9
_STEP_MIN_FACTOR = 0.2
_STEP_MAX_FACTOR = 5.0
_MAX_STEP_REJECTIONS = 12


@dataclass
class ODEState:
    """State container for batched ODE integration."""
    y: np.ndarray           # (n_cells, n_species) current state
    t: float                # current time
    dt: float               # current step size
    order: int              # current BDF/Adams order (1-5)
    n_steps: int = 0        # steps taken
    n_rhs_evals: int = 0    # RHS evaluations
    n_jac_evals: int = 0    # Jacobian evaluations
    history: Optional[np.ndarray] = None  # Nordsieck array for multistep
    n_rejected: int = 0     # steps rejected by the error controller
    # Scaled error norm of the last accepted step, for the PI controller.
    err_prev: float = 1.0


@dataclass
class ODESolution:
    """Result of ODE integration."""
    t: np.ndarray           # (n_times,) time points
    y: np.ndarray           # (n_times, n_cells, n_species) solution
    success: bool
    message: str
    n_steps: int
    n_rhs_evals: int


# ── Batched ODE Integrator ───────────────────────────────────────────────

class BatchedODEIntegrator:
    """
    GPU-accelerated batched ODE integrator.

    Solves the same ODE system for many cells in parallel, where each
    cell can have different parameters and initial conditions.

    Parameters
    ----------
    system : ODESystem
        ODE system definition
    n_cells : int
        Number of cells to simulate in parallel
    method : str
        Integration method: 'bdf' (stiff), 'adams' (non-stiff), 'rk45' (explicit)
    rtol : float
        Relative tolerance for adaptive stepping
    atol : float
        Absolute tolerance for adaptive stepping
    max_order : int
        Maximum order for BDF/Adams methods (1-5)

    Examples
    --------
    >>> system = ODESystem.gene_expression_2species()
    >>> solver = BatchedODEIntegrator(system, n_cells=10000)
    >>>
    >>> # Set initial conditions (10 mRNA, 100 protein per cell)
    >>> y0 = np.tile([10.0, 100.0], (10000, 1))
    >>>
    >>> # Integrate from t=0 to t=10 hours
    >>> solution = solver.integrate(t_span=(0, 10), y0=y0)
    >>>
    >>> # Get mean protein levels at final time
    >>> print(f"Mean protein: {solution.y[-1, :, 1].mean():.1f}")
    """

    def __init__(
        self,
        system: ODESystem,
        n_cells: int,
        method: str = 'bdf',
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_order: int = 5,
    ):
        self.system = system
        self.n_cells = n_cells
        self.method = method.lower()
        self.rtol = rtol
        self.atol = atol
        self.max_order = min(max_order, 5)

        self._backend = get_backend()
        self._state: Optional[ODEState] = None

        # Per-cell parameters: (n_cells, n_params)
        # Default: replicate system parameters for all cells
        self._cell_params: Optional[np.ndarray] = None

        # Compiled GPU kernels
        self._rhs_kernel = None
        self._jac_kernel = None
        self._newton_kernel = None
        self._error_kernel = None

        if self._backend.has_gpu:
            self._compile_kernels()

        log.info(f"BatchedODEIntegrator: {n_cells} cells, method={method}, "
                 f"GPU={'yes' if self._backend.has_gpu else 'no'}")

    def _compile_kernels(self):
        """Compile CUDA kernels for GPU acceleration."""
        try:
            import cupy as cp
            self._rhs_kernel = cp.RawKernel(_BDF_RHS_KERNEL, "bdf_rhs_evaluation")
            self._jac_kernel = cp.RawKernel(_BDF_JACOBIAN_KERNEL, "bdf_jacobian_eval")
            self._newton_kernel = cp.RawKernel(_BDF_NEWTON_KERNEL, "bdf_newton_step")
            self._error_kernel = cp.RawKernel(_ERROR_ESTIMATION_KERNEL, "estimate_error")
            log.debug("ODE CUDA kernels compiled successfully")
        except Exception as e:
            log.warning(f"Failed to compile ODE CUDA kernels: {e}")

    def set_cell_parameters(self, cell_params: np.ndarray):
        """
        Set per-cell parameters.

        Parameters
        ----------
        cell_params : np.ndarray
            (n_cells, n_params) array of parameters per cell
        """
        self._cell_params = cell_params.astype(np.float32)

    def _get_cell_params(self) -> np.ndarray:
        """Get per-cell parameters, initializing from system params if needed."""
        if self._cell_params is not None:
            return self._cell_params

        # Create default from system parameters
        param_values = list(self.system.parameters.values())
        self._cell_params = np.tile(
            np.array(param_values, dtype=np.float32),
            (self.n_cells, 1)
        )
        return self._cell_params

    def _param_dict(self) -> Dict[str, Any]:
        """
        Parameters for rhs_func/jacobian_func evaluation.

        When per-cell parameters are set, each value is an (n_cells,) column
        rather than a scalar. Both rhs and jacobian index params by name and
        combine them with (n_cells,)-shaped species slices, so the columns
        broadcast without any change to model definitions.
        """
        if self._cell_params is None:
            return self.system.parameters
        return {
            name: self._cell_params[:, i]
            for i, name in enumerate(self.system.parameters)
        }

    def integrate(
        self,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None,
        max_steps: int = 100000,
    ) -> ODESolution:
        """
        Integrate the ODE system.

        Parameters
        ----------
        t_span : tuple
            (t_start, t_end) integration interval
        y0 : np.ndarray
            (n_cells, n_species) initial conditions
        t_eval : np.ndarray, optional
            Times at which to store solution (if None, adaptive)
        max_steps : int
            Maximum number of integration steps

        Returns
        -------
        ODESolution
            Solution object with time points and state trajectory
        """
        t0, tf = t_span

        # Initialize state. float64 input is preserved rather than
        # downcast: single precision puts a ~1e-7 floor under the
        # solution, so tightening rtol below that bought nothing and the
        # error actually grew again from round-off. float32 callers (the
        # GPU BDF path) keep the dtype they passed.
        y0 = np.asarray(y0)
        state_dtype = np.float64 if y0.dtype == np.float64 else np.float32
        self._state = ODEState(
            y=y0.astype(state_dtype),
            t=t0,
            dt=min(0.01, (tf - t0) / 100),  # Initial step size
            order=1,
        )

        # Storage for solution
        if t_eval is not None:
            t_out = t_eval
        else:
            # Adaptive output: store ~100 points
            n_out = min(100, max_steps)
            t_out = np.linspace(t0, tf, n_out)

        y_out = []
        t_stored = []

        y_out.append(self._state.y.copy())
        t_stored.append(t0)

        next_out_idx = 1

        # Integration loop
        while self._state.t < tf and self._state.n_steps < max_steps:
            # Never step past the end of the interval. The loop exits on
            # `t >= tf` but nothing clamped the final step, so the solution
            # was reported at whatever time the last step happened to land
            # on. With a fixed small dt that was a small overshoot; once
            # the step size adapts and can grow 5x per step it becomes
            # large -- integrating dy/dt=-2y to t=2 actually stopped at
            # t=2.26 and reported that as the answer.
            remaining = tf - self._state.t
            suggested_dt = self._state.dt
            if suggested_dt > remaining:
                self._state.dt = remaining

            # Take one step
            if self.method == 'bdf':
                self._step_bdf()
            elif self.method == 'adams':
                self._step_adams()
            else:
                self._step_rk45()

            # Restore the controller's own suggestion if it was only
            # clamped to fit the interval, so the clamp does not ratchet
            # the step size down for the rest of the run.
            if suggested_dt > remaining:
                self._state.dt = max(self._state.dt, suggested_dt)

            # Store output at requested times
            while next_out_idx < len(t_out) and self._state.t >= t_out[next_out_idx]:
                # Interpolate if needed (for simplicity, just use current)
                y_out.append(self._state.y.copy())
                t_stored.append(t_out[next_out_idx])
                next_out_idx += 1

        return ODESolution(
            t=np.array(t_stored),
            y=np.array(y_out),
            success=self._state.t >= tf,
            message="Integration complete" if self._state.t >= tf else "Max steps reached",
            n_steps=self._state.n_steps,
            n_rhs_evals=self._state.n_rhs_evals,
        )

    def step(self, dt: float):
        """
        Take a single integration step of size dt.

        May subdivide dt internally for accuracy.
        """
        if self._state is None:
            raise RuntimeError("Integrator not initialized. Call integrate() first.")

        t_end = self._state.t + dt
        while self._state.t < t_end:
            self._state.dt = min(self._state.dt, t_end - self._state.t)
            if self.method == 'bdf':
                self._step_bdf()
            elif self.method == 'adams':
                self._step_adams()
            else:
                self._step_rk45()

    def _step_bdf(self):
        """BDF (implicit) step for stiff systems."""
        if self._backend.has_gpu and self._newton_kernel is not None:
            self._step_bdf_gpu()
        else:
            self._step_bdf_cpu()

    def _step_bdf_gpu(self):
        """GPU BDF step using CUDA kernels.

        The CUDA kernels here are hardcoded to the 2-species gene
        expression model: ``_BDF_RHS_KERNEL`` reads ``y[base+0]`` and
        ``y[base+1]`` only, and ``_BDF_NEWTON_KERNEL``'s solve is guarded
        by ``if (n_species == 2)`` with no else branch. For any other
        system the kernels wrote nothing and the step returned silently
        having done *no integration at all* -- so, for example,
        ``create_ode_solver('ar_signaling', method='bdf')`` on a GPU
        produced a flat trajectory that looked like a converged steady
        state. ``system.rhs_func`` is never consulted on this path.

        Until the kernels carry a general LU solve, anything that is not
        the 2-species model goes to the CPU BDF path, which does use
        ``rhs_func`` and does integrate.
        """
        import cupy as cp

        if self.system.n_species != 2:
            if not getattr(self, "_bdf_gpu_fallback_warned", False):
                log.warning(
                    "GPU BDF kernels only implement the 2-species model; "
                    "this system has %d species (%s). Falling back to the "
                    "CPU BDF path so the system is actually integrated.",
                    self.system.n_species,
                    ", ".join(self.system.species_names),
                )
                self._bdf_gpu_fallback_warned = True
            self._step_bdf_cpu()
            return

        # The kernels are compiled against `float*`, so the state must be
        # single precision here regardless of what the caller supplied.
        y = cp.asarray(self._state.y, dtype=cp.float32)
        cell_params = cp.asarray(self._get_cell_params())
        # The RHS kernel reads its rate constants from `cell_params`, not
        # from this array -- it is passed only to satisfy the signature.
        params = cp.asarray(list(self.system.parameters.values()), dtype=np.float32)

        n_cells, n_species = y.shape
        n_params = len(params)
        n_cell_params = cell_params.shape[1]

        # BDF-1 (backward Euler) for simplicity
        # y_{n+1} = y_n + dt * f(y_{n+1})
        # Solve via Newton iteration

        gamma = 1.0  # BDF-1 coefficient
        dt = self._state.dt

        # Predictor: y_pred = y_n
        y_pred = y.copy()
        y_new = y.copy()

        f = cp.zeros_like(y)
        J = cp.zeros((n_cells, n_species, n_species), dtype=np.float32)

        block = 256
        grid = (n_cells + block - 1) // block

        # Newton iterations
        for _ in range(3):
            # Evaluate RHS
            self._rhs_kernel(
                (grid,), (block,),
                (y_new, f, params, cell_params,
                 np.int32(n_cells), np.int32(n_species),
                 np.int32(n_params), np.int32(n_cell_params))
            )

            # Evaluate Jacobian
            self._jac_kernel(
                (grid,), (block,),
                (y_new, J, cell_params,
                 np.int32(n_cells), np.int32(n_species), np.int32(n_cell_params))
            )

            # Newton step
            self._newton_kernel(
                (grid,), (block,),
                (y_new, y_pred, f, J,
                 np.float32(dt), np.float32(gamma),
                 np.int32(n_cells), np.int32(n_species))
            )

        cp.cuda.Stream.null.synchronize()

        # Update state
        self._state.y = cp.asnumpy(y_new)
        self._state.t += dt
        self._state.n_steps += 1
        self._state.n_rhs_evals += 3
        self._state.n_jac_evals += 3

    def _step_bdf_cpu(self):
        """CPU BDF step using scipy or manual Newton."""
        from scipy.optimize import fsolve

        y = self._state.y
        dt = self._state.dt
        params = self._param_dict()

        def residual(y_new_flat):
            y_new = y_new_flat.reshape(y.shape)
            f = self.system.rhs_func(self._state.t + dt, y_new, params)
            return (y_new - y - dt * f).flatten()

        y_new_flat = fsolve(residual, y.flatten(), full_output=False)
        self._state.y = y_new_flat.reshape(y.shape)
        self._state.t += dt
        self._state.n_steps += 1
        self._state.n_rhs_evals += 10  # Approximate

    def _step_adams(self):
        """Adams-Moulton is not implemented; runs adaptive RK45 instead.

        This used to call ``_step_bdf()``, which is backward Euler -- a
        first-order *implicit* method. Requesting a non-stiff integrator
        and silently receiving the stiff one is the wrong direction twice
        over, and it was undocumented.

        Adaptive Dormand-Prince is the closest honest substitute: it is
        explicit, it is what a non-stiff problem wants, and it is error
        controlled. The substitution is announced once per integrator so
        it cannot pass unnoticed.
        """
        if not getattr(self, "_adams_warned", False):
            logger.warning(
                "method='adams': Adams-Moulton is not implemented. Using "
                "adaptive Dormand-Prince RK45, which is also a non-stiff "
                "method. Pass method='rk45' to select it directly, or "
                "method='bdf' if the system is stiff."
            )
            self._adams_warned = True
        self._step_rk45()

    def _step_rk45(self):
        """Explicit RK45 step."""
        if self._backend.has_gpu:
            self._step_rk45_gpu()
        else:
            self._step_rk45_cpu()

    def _step_rk45_gpu(self):
        """RK45 step.

        There is no separate GPU path. ``system.rhs_func`` is a NumPy
        callable, so the previous "GPU" implementation copied the state
        device->host and back on *every* stage -- four full round-trips per
        step, which made it slower than the CPU path while looking like an
        acceleration. Running the explicit stages in NumPy is both faster
        and honest. The GPU is used where it actually helps: the BDF
        Newton kernels.
        """
        self._step_rk45_cpu()

    def _step_rk45_cpu(self):
        """Adaptive Dormand-Prince 5(4) step with PI step-size control.

        Computes a 5th-order solution and an embedded 4th-order solution
        from the same stages. Their difference estimates the local error,
        which is compared against ``atol + rtol*|y|``; the step is retried
        with a smaller ``dt`` if it fails and ``dt`` grows when there is
        headroom.

        This replaces a fixed-step classical RK4 that ignored ``rtol`` and
        ``atol`` entirely -- they were stored on the integrator and never
        read, so requesting a tighter tolerance changed nothing about the
        answer, and there was no error estimate to control a step size
        with.

        The error norm is taken over the whole batch, so all cells share
        one time step. That is what the batched design requires: cells
        advance together and a per-cell dt would desynchronise them.
        """
        params = self._param_dict()
        rhs = self.system.rhs_func
        t = self._state.t
        y = self._state.y
        n_stages = len(_DP_C)

        dt = float(self._state.dt)
        if dt <= 0.0:
            raise ValueError(f"Step size must be positive, got {dt}")

        for _ in range(_MAX_STEP_REJECTIONS):
            k = np.empty((n_stages,) + y.shape, dtype=np.float64)
            k[0] = rhs(t, y, params)

            for i in range(1, n_stages):
                y_stage = y.astype(np.float64)
                for j, a_ij in enumerate(_DP_A[i]):
                    if a_ij:
                        y_stage = y_stage + (dt * a_ij) * k[j]
                k[i] = rhs(t + _DP_C[i] * dt, y_stage, params)

            self._state.n_rhs_evals += n_stages

            y_new = y.astype(np.float64) + dt * np.tensordot(_DP_B5, k, axes=(0, 0))
            err_vec = dt * np.tensordot(_DP_E, k, axes=(0, 0))

            # Componentwise tolerance, then an RMS norm over the batch.
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_new))
            err_norm = float(np.sqrt(np.mean((err_vec / scale) ** 2)))

            if not np.isfinite(err_norm):
                # A blown-up stage: shrink hard rather than propagate NaN.
                dt *= _STEP_MIN_FACTOR
                self._state.n_rejected += 1
                continue

            if err_norm <= 1.0:
                # Accept.
                self._state.y = y_new.astype(y.dtype)
                self._state.t = t + dt
                self._state.n_steps += 1

                factor = _STEP_SAFETY * (
                    max(err_norm, 1e-10) ** -_PI_ALPHA
                ) * (max(self._state.err_prev, 1e-10) ** _PI_BETA)
                factor = min(_STEP_MAX_FACTOR, max(_STEP_MIN_FACTOR, factor))
                self._state.dt = dt * factor
                self._state.err_prev = max(err_norm, 1e-4)
                return

            # Reject: retry the same step with a smaller dt. A rejected
            # step must not advance t -- that is the difference between
            # error control and simply taking a wrong step anyway.
            self._state.n_rejected += 1
            factor = max(
                _STEP_MIN_FACTOR, _STEP_SAFETY * err_norm ** -_PI_ALPHA
            )
            dt *= factor

        raise RuntimeError(
            f"RK45 could not meet rtol={self.rtol:g}/atol={self.atol:g} after "
            f"{_MAX_STEP_REJECTIONS} step reductions (dt={dt:g}). The system "
            f"is likely stiff -- use method='bdf'."
        )

    def get_state(self) -> np.ndarray:
        """Get current state (n_cells, n_species)."""
        if self._state is None:
            raise RuntimeError("Integrator not initialized")
        return self._state.y.copy()

    def get_species(self, name: str) -> np.ndarray:
        """Get values for a specific species across all cells."""
        if self._state is None:
            raise RuntimeError("Integrator not initialized")
        idx = self.system.species_names.index(name)
        return self._state.y[:, idx].copy()


# ── Physics Model Registration ───────────────────────────────────────────

@register_physics(
    "cupy_ode_adaptive",
    version="1.0.0",
    backend=PhysicsBackendType.CUPY,
    model_type=PhysicsModelType.ODE,
)
class CuPyAdaptiveODESolver(BasePhysicsModel):
    """
    GPU-accelerated adaptive ODE solver following PhysicsModel protocol.

    Wraps BatchedODEIntegrator for use with the physics registry system.
    """

    def __init__(self, name: str = "cupy_ode_adaptive", config: Optional[Dict] = None):
        super().__init__(name, config)
        self._integrator: Optional[BatchedODEIntegrator] = None
        self._system: Optional[ODESystem] = None

    @property
    def backend_type(self) -> PhysicsBackendType:
        return PhysicsBackendType.CUPY

    @property
    def model_type(self) -> PhysicsModelType:
        return PhysicsModelType.ODE

    def initialize(self, state: PhysicsState, **config) -> None:
        """
        Initialize the ODE solver.

        Config options:
            system: ODESystem instance or 'gene_expression' / 'ar_signaling'
            n_cells: Number of cells (default: from state.n_particles)
            method: 'bdf', 'adams', or 'rk45'
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        # Get or create ODE system
        system_config = config.get('system', 'gene_expression')
        if isinstance(system_config, ODESystem):
            self._system = system_config
        elif system_config == 'ar_signaling':
            self._system = ODESystem.ar_signaling_pathway()
        else:
            self._system = ODESystem.gene_expression_2species()

        n_cells = config.get('n_cells', state.n_particles or 1000)
        method = config.get('method', 'bdf' if self._system.stiff else 'rk45')
        rtol = config.get('rtol', 1e-3)
        atol = config.get('atol', 1e-6)

        self._integrator = BatchedODEIntegrator(
            system=self._system,
            n_cells=n_cells,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        # Initialize ODE state from PhysicsState custom data
        if 'ode_y' in state.custom:
            y0 = state.custom['ode_y']
        else:
            # Default initial conditions
            y0 = np.zeros((n_cells, self._system.n_species), dtype=np.float32)
            y0[:, 0] = 10.0  # Initial mRNA

        self._integrator._state = ODEState(
            y=y0,
            t=config.get('t0', 0.0),
            dt=config.get('dt', 0.01),
            order=1,
        )

        self._initialized = True
        log.info(f"ODE solver initialized: {n_cells} cells, "
                 f"{self._system.n_species} species, method={method}")

    def step(self, dt: float, state: PhysicsState) -> PhysicsState:
        """Advance ODE system by dt."""
        if not self._initialized or self._integrator is None:
            raise RuntimeError("ODE solver not initialized")

        self._integrator.step(dt)

        # Store result in PhysicsState
        state.custom['ode_y'] = self._integrator.get_state()
        state.custom['ode_t'] = self._integrator._state.t

        return state

    def get_species(self, name: str) -> np.ndarray:
        """Get values for a specific species."""
        if self._integrator is None:
            raise RuntimeError("ODE solver not initialized")
        return self._integrator.get_species(name)


# ── Convenience Functions ────────────────────────────────────────────────

def create_ode_solver(
    system: Union[str, ODESystem] = 'gene_expression',
    n_cells: int = 1000,
    method: str = 'bdf',
    **kwargs
) -> BatchedODEIntegrator:
    """
    Create a batched ODE solver.

    Parameters
    ----------
    system : str or ODESystem
        'gene_expression', 'ar_signaling', or ODESystem instance
    n_cells : int
        Number of cells to simulate
    method : str
        Integration method: 'bdf', 'adams', 'rk45'
    **kwargs
        Additional arguments passed to BatchedODEIntegrator

    Returns
    -------
    BatchedODEIntegrator
        Configured solver instance
    """
    if isinstance(system, str):
        if system == 'ar_signaling':
            system = ODESystem.ar_signaling_pathway()
        else:
            system = ODESystem.gene_expression_2species()

    return BatchedODEIntegrator(system, n_cells, method, **kwargs)
