"""Where a population starts in its cell cycle.

Both engines used to start every cell at age 0 with its lineage's division
time, so a whole lineage divided on one step, and whether that step fell
inside a run depended on whether the float sum of the steps reached the
division time exactly. Starting from the steady-state age distribution
spreads divisions through the run, as in a real tissue.
"""
from __future__ import annotations

import numpy as np


def steady_state_ages(periods, uniforms):
    """Ages from the stable age distribution of balanced exponential growth.

    With every cell dividing at age T into two cells of age 0, a population
    grows as 2^(t/T), so the cells now of age a were born when it was
    2^(-a/T) of its present size: density (2 ln 2 / T) 2^(-a/T) on [0, T).
    A cycling population settles into this distribution, so a run that
    starts from it starts at steady state. Sampled by inverting the CDF,
    F(a) = 2 (1 - 2^(-a/T)).

    ``periods`` is each cell's division time and ``uniforms`` one U[0, 1)
    draw per cell, so the caller keeps control of its random stream. The
    mean age is (2 - 2 ln 2 - ... ) -- numerically 0.4427 T -- and 58.6% of
    cells are younger than T/2.
    """
    periods = np.asarray(periods, dtype=float)
    uniforms = np.asarray(uniforms, dtype=float)
    return -periods * np.log2(1.0 - uniforms / 2.0)
