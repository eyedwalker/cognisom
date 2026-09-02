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
