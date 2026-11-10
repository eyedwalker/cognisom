"""An engine run caps BLAS for its own length and gives the pool back.

Measured 2026-11-10 on the demo's run: one thread returns the same time
series as a thread per core, sooner, on about a tenth of the CPU. The
claim tested here is the mechanism -- the cap is on DURING the run, off
after it, and can be declined -- not the timing.
"""

import contextlib
import io

import pytest

from cognisom.dashboard import engine_runner as er

threadpoolctl = pytest.importorskip("threadpoolctl")


def _blas_threads():
    import numpy as np
    np.ones((4, 4)) @ np.ones((4, 4))
    return [p["num_threads"] for p in threadpoolctl.threadpool_info()
            if p.get("user_api") == "blas"]


def test_the_cap_holds_inside_and_is_restored_after(monkeypatch):
    monkeypatch.delenv(er.ENGINE_BLAS_THREADS_ENV, raising=False)
    with threadpoolctl.threadpool_limits(limits=3, user_api="blas"):
        before = _blas_threads()
        if not before or max(before) < 2:
            pytest.skip("this machine's BLAS cannot run two threads")
        with er.engine_blas_threads():
            assert set(_blas_threads()) == {er.ENGINE_BLAS_THREADS_DEFAULT}
        assert _blas_threads() == before, "the run must give the pool back"


def test_zero_leaves_the_pool_alone(monkeypatch):
    monkeypatch.setenv(er.ENGINE_BLAS_THREADS_ENV, "0")
    before = _blas_threads()
    with er.engine_blas_threads():
        assert _blas_threads() == before


def test_a_setting_that_is_not_a_number_is_refused(monkeypatch):
    monkeypatch.setenv(er.ENGINE_BLAS_THREADS_ENV, "one")
    with pytest.raises(ValueError, match="not a whole number"):
        er.engine_blas_threads()


def test_run_itself_goes_through_the_cap(monkeypatch):
    """The helper existing is not the run using it."""
    seen = []
    real = er.engine_blas_threads

    def spy():
        seen.append(True)
        return real()

    monkeypatch.setattr(er, "engine_blas_threads", spy)
    runner = er.EngineRunner(tissue="ring", dt=0.05, duration=0.1)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    assert seen == [True]
