"""
Test configuration for the root suite.

Neoantigen binding prediction now refuses to fall back to the approximate
position-weight-matrix scorer unless COGNISOM_ALLOW_PWM_FALLBACK is set,
because a silent downgrade is what let the PWM run unnoticed in production.

Integration tests that merely need *a* binding number (closed-loop event
tracing, peptide plumbing) should still be runnable on a machine without
MHCflurry. This hook grants that permission explicitly and announces it, so
a run against the approximation can never be mistaken for a run against the
real predictor -- which is the distinction the silent fallback destroyed.

CI and the production image are Python 3.11 with mhcflurry installed, so the
banner below does not appear there and the real path is what gets tested.
"""
from __future__ import annotations

import os

# One BLAS thread per test process.
#
# OpenBLAS starts a thread per core. This suite is run under xdist, so on
# an 18-core laptop four workers started 72 spinning threads; measured on
# one slow file, 3,990 s of SYSTEM time against 11 s capped, and the run
# was a quarter slower even alone. Under the full suite the load average
# reached 60-77 and tests that take 40 s died at the 120 s timeout with
# nothing wrong with them. The engine's arrays are small enough that the
# threads only contend.
#
# setdefault, so a caller can still ask for more.
#
# The environment variable alone is NOT enough. It is read when the BLAS
# library loads, and the zarr pytest plugin imports numpy before this
# file runs -- so it reaches xdist workers (they inherit the environment)
# and silently misses a plain single-process run. The pool that is
# already loaded is therefore capped at runtime as well.
# test_the_suite_does_not_oversubscribe_the_machine checks the pool
# itself, in whichever kind of process it runs in.
_BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                     "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_NUM_THREADS")
_caller_chose_threads = any(v in os.environ for v in _BLAS_THREAD_VARS)
for _var in _BLAS_THREAD_VARS:
    os.environ.setdefault(_var, "1")
if not _caller_chose_threads:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        pass


def _mhcflurry_usable() -> bool:
    """Whether MHCflurry can actually predict, not merely be imported.

    The package installing and the ~600 MB model weights being present are
    separate facts: `pip install mhcflurry` succeeds on its own, and
    Class1PresentationPredictor.load() then fails with "Missing MHCflurry
    downloadable file". Checking only importability reported the real scorer
    as active on a machine that could not run it.
    """
    try:
        from mhcflurry import Class1PresentationPredictor
        Class1PresentationPredictor.load()
        return True
    except Exception:
        return False


def pytest_configure(config):
    if _mhcflurry_usable():
        config.stash["cognisom_binding_scorer"] = "mhcflurry"
        return

    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    config.stash["cognisom_binding_scorer"] = "pwm-fallback"


def pytest_report_header(config):
    scorer = config.stash.get("cognisom_binding_scorer", "unknown")
    if scorer == "mhcflurry":
        return "binding scorer: MHCflurry (real predictor)"
    return (
        "binding scorer: PWM FALLBACK -- MHCflurry is not usable here (not "
        "installed, or installed without its model weights), so affinity-"
        "dependent assertions exercise the approximation, not the shipped "
        "predictor. mhcflurry 2.0.6 requires Python <= 3.12, and its weights "
        "come from `mhcflurry-downloads fetch models_class1_presentation`."
    )
