"""The cap in conftest.py has to actually reach the BLAS library.

An environment variable set after numpy has loaded changes nothing and
reports nothing, so the cap is checked where it matters: on the thread
pool numpy is really using in this process.
"""

import os

import pytest


def test_blas_runs_one_thread_per_test_process():
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        pytest.skip("the caller asked for a different thread count")
    threadpoolctl = pytest.importorskip("threadpoolctl")
    import numpy as np

    np.ones((8, 8)) @ np.ones((8, 8))
    pools = [p for p in threadpoolctl.threadpool_info()
             if p.get("user_api") == "blas"]
    assert pools, "numpy reported no BLAS pool to check"
    wide = {p.get("internal_api"): p["num_threads"]
            for p in pools if p["num_threads"] != 1}
    assert not wide, (
        "BLAS started more than one thread per test process, so the cap "
        "in tests/conftest.py ran after numpy was imported", wide)
