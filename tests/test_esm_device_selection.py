"""The ESM-2 scorer must use the machine's GPU, and say which one it picked.

`RealESMStabilityScorer` loaded a 150M-parameter masked-LM and never called
`.to(device)`, so every score ran on the CPU no matter what the host had. On an
Apple Silicon workstation that left a 40-core GPU idle: measured on an M5 Max
scoring TP53 R175 with the default 150M model, 132.0 ms/score on CPU against
22.6 ms on MPS.

The ordering matters as much as the fact of choosing. CUDA stays first because
the production GPU box has one and it is still the fastest path; MPS is the
fallback that makes a Mac use its GPU rather than silently not. These tests
drive `_select_device` with stand-in torch objects so the precedence is pinned
without needing any particular hardware present -- the real devices cannot be
summoned on demand in CI, which is exactly why this was never noticed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cognisom.engine.py.molecular.esm_stability import _select_device


def _torch(*, cuda: bool, mps: bool | None) -> SimpleNamespace:
    """A stand-in exposing only what _select_device probes.

    `mps=None` models an older torch with no `backends.mps` attribute at all,
    which must not raise.
    """
    backends = SimpleNamespace()
    if mps is not None:
        backends.mps = SimpleNamespace(is_available=lambda: mps)
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda), backends=backends)


def test_cuda_wins_when_present():
    """Production has CUDA and it stays the fastest path."""
    assert _select_device(_torch(cuda=True, mps=False)) == "cuda"


def test_cuda_still_wins_even_if_mps_also_reports_available():
    assert _select_device(_torch(cuda=True, mps=True)) == "cuda"


def test_mps_is_used_when_there_is_no_cuda():
    """The Apple Silicon case: the regression this whole module had."""
    assert _select_device(_torch(cuda=False, mps=True)) == "mps"


def test_cpu_when_no_accelerator_is_available():
    assert _select_device(_torch(cuda=False, mps=False)) == "cpu"


def test_a_torch_without_an_mps_backend_does_not_explode():
    """`torch.backends.mps` did not always exist; probing must stay safe."""
    assert _select_device(_torch(cuda=False, mps=None)) == "cpu"


@pytest.mark.parametrize("requested", ["cpu", "cuda", "mps"])
def test_an_explicit_request_overrides_the_probe(requested):
    """Forcing a device must win, including forcing CPU on a GPU machine.

    MPS and CPU do not agree bit-for-bit -- worst measured disagreement in
    delta_log_likelihood over three TP53 hotspots was 6.3e-06 -- so a caller
    who needs reproducible output has to be able to pin the backend.
    """
    assert _select_device(_torch(cuda=True, mps=True), requested) == requested
