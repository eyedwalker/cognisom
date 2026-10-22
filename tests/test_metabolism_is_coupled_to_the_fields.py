"""ATP and lactate come from the FBA model, and the model had to be fixed first.

``cognisom.gpu.fba_solver`` held a metabolic model that nothing used, and it
could not have been used as it stood:

* ``oxidative_phosph`` was written as ``O2 -> 2.5 ATP``, producing ATP from
  oxygen with no carbon source and no reducing equivalents. A cell with oxygen
  and zero glucose therefore grew at the maximum rate, and since oxygen alone
  sufficed, glycolysis and the TCA cycle went unused whenever oxygen was
  present -- so lactate production was always exactly zero and the model could
  not express the Warburg effect it exists to study.

* Lactate and CO2 were dead ends: produced by reactions, exported by none.
  Under the steady-state constraint S*v = 0 that forces both producing
  reactions to zero, which starves pyruvate and NADH of any sink and shuts
  every carbon route. The unphysical O2 -> ATP reaction was load-bearing -- it
  was the only feasible way to make ATP, kept feasible precisely by producing
  nothing that needed disposing of. Correcting the stoichiometry without
  adding the exchanges makes the whole model infeasible, which is how the
  omission surfaced.

* An infeasible LP returned ``(lb + ub) / 2``. That vector does not satisfy
  S*v = 0, so it is not a flux distribution; with oxygen and glucose both
  clamped to zero it reported a growing cell excreting lactate. Infeasibility
  is the most consequential thing this model can say about a cell in a
  hypoxic core, and it was being converted into fabricated growth.

With NADH carrying the oxidative yield, the glycolytic shift is no longer
something to impose: without oxygen the only NADH sink is lactate, which caps
glycolysis at 2 ATP per glucose and forces fermentation.
"""

import numpy as np
import pytest

from cognisom.core import metabolism
from cognisom.gpu.fba_solver import FBASolver, prostate_cancer_model


@pytest.fixture(scope="module")
def model():
    return prostate_cancer_model()


@pytest.fixture(scope="module")
def solver(model):
    return FBASolver(model, use_gpu=False)


def _at(solver, model, o2, glucose, objective=None):
    lb, ub = metabolism._bounds_for(model, o2, glucose)
    if objective is not None:
        model = type(model)(
            S=model.S, lb=model.lb, ub=model.ub, objective=objective,
            metabolite_names=list(model.metabolite_names),
            reaction_names=list(model.reaction_names))
        solver = FBASolver(model, use_gpu=False)
    return solver.solve_single(lb, ub)


def test_every_metabolite_can_leave(model):
    """A produced metabolite with no sink forces its producer to zero."""
    produced = model.S > 0
    consumed = model.S < 0
    for row, name in enumerate(model.metabolite_names):
        if not produced[row].any():
            continue
        makers = [model.reaction_names[j]
                  for j in np.flatnonzero(produced[row])]
        assert consumed[row].any(), (
            f"{name} is produced by {makers} and consumed by nothing, so "
            "S*v = 0 pins every one of those reactions at zero"
        )


def test_atp_needs_a_carbon_source(model, solver):
    """Oxygen alone must not make ATP. This was the central defect."""
    assert _at(solver, model, 0.21, 0.0) is None, (
        "a cell with oxygen and no glucose must be infeasible: there is "
        "nothing to oxidise"
    )
    assert _at(solver, model, 0.0, 0.0) is None
    # And with a carbon source it is feasible.
    assert _at(solver, model, 0.21, 5.0) is not None


def test_the_yields_are_the_textbook_ones(model, solver):
    """Anaerobic glycolysis gives exactly 2 lactate per glucose.

    Checked as a ratio of fluxes at zero oxygen, where the only route is
    fermentation, so it is a statement about the stoichiometry and not about
    the capacities.
    """
    fluxes = _at(solver, model, 0.0, 5.0, objective=metabolism._atp_objective(model))
    assert fluxes is not None
    glucose_in = fluxes["glucose_uptake"]
    lactate_out = fluxes["lactate_export"]
    assert glucose_in > 0
    assert lactate_out / glucose_in == pytest.approx(2.0, rel=1e-6), (
        f"{lactate_out:.3f} lactate per {glucose_in:.3f} glucose"
    )
    # 2 ATP per glucose, from substrate-level phosphorylation only.
    atp = float(np.dot(metabolism._atp_objective(model),
                       [fluxes[n] for n in model.reaction_names]))
    assert atp / glucose_in == pytest.approx(2.0, rel=1e-6)


def test_fluxes_balance(model, solver):
    """S*v = 0, which the midpoint fallback never satisfied."""
    for o2, glucose in ((0.21, 5.0), (0.001, 5.0), (0.0, 5.0), (0.21, 1.0)):
        fluxes = _at(solver, model, o2, glucose)
        if fluxes is None:
            continue
        v = np.array([fluxes[n] for n in model.reaction_names])
        assert np.abs(model.S.dot(v)).max() < 1e-9, (o2, glucose)


def test_an_infeasible_problem_returns_nothing_rather_than_a_vector(model):
    """None, not a plausible-looking flux distribution."""
    solver = FBASolver(model, use_gpu=False)
    lb, ub = model.lb.copy(), model.ub.copy()
    ub[model.reaction_names.index("glucose_uptake")] = 0.0
    ub[model.reaction_names.index("oxygen_consumption")] = 0.0
    assert solver.solve_single(lb, ub) is None

    fluxes, feasible = solver.solve_batch(
        3, np.tile(lb, (3, 1)), np.tile(ub, (3, 1)))
    assert not feasible.any()
    assert np.all(fluxes == 0.0), (
        "an infeasible row must be zero AND flagged; a reader who ignores "
        "the flag should get an obviously empty answer, not a growing cell"
    )


# ── the coupling ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def table():
    return metabolism.table()


def test_the_table_agrees_with_a_direct_solve_off_grid(table, model):
    """Interpolation is the only approximation in the coupling, so bound it.

    Points chosen BETWEEN grid nodes, since on a node the table is exact by
    construction and would prove nothing.
    """
    objective = metabolism._atp_objective(model)
    solver = FBASolver(
        type(model)(S=model.S, lb=model.lb, ub=model.ub, objective=objective,
                    metabolite_names=list(model.metabolite_names),
                    reaction_names=list(model.reaction_names)),
        use_gpu=False)
    reference = float(np.dot(
        objective,
        [_at(solver, model, metabolism.O2_REFERENCE,
             metabolism.GLUCOSE_REFERENCE, objective)[n]
         for n in model.reaction_names]))

    def error_at(o2):
        glucose = metabolism.GLUCOSE_REFERENCE
        got, _, feasible = metabolism.evaluate(
            np.array([o2]), np.array([glucose]), table)
        direct = _at(solver, model, o2, glucose, objective)
        if direct is None or not feasible[0]:
            return None
        exact = float(np.dot(objective,
                             [direct[n] for n in model.reaction_names]))
        return abs(got[0] - exact / reference)

    midpoints = [0.5 * (table.o2_levels[i] + table.o2_levels[i + 1])
                 for i in range(len(table.o2_levels) - 1)]
    errors = [error_at(o2) for o2 in midpoints]

    # Everywhere except the interval touching zero oxygen, interpolation is
    # tight.
    away = [e for e in errors[1:] if e is not None]
    assert away and max(away) < 0.02, (
        f"interpolation error {max(away):.3f} in ATP charge away from the "
        "step at zero oxygen"
    )

    # The first interval is worse, and for a reason no grid can fix: the
    # model has a genuine step at zero oxygen. With no oxygen at all,
    # fermentation alone caps ATP production at 2 per glucose, which is 20
    # against an ATP disposal capacity of 25, so the charge is 0.8. With ANY
    # oxygen the cell runs enough oxidative phosphorylation to reach the cap
    # and the charge is 1.0. Linear interpolation across a step is wrong by
    # half its height at the midpoint, and refining the grid only narrows the
    # interval the error lives in -- it never removes it.
    first = errors[0]
    assert first is not None and first > 2 * max(away), (
        "the step at zero oxygen should dominate the interpolation error; if "
        "it no longer does, this explanation is stale"
    )
    assert first < 0.15, f"but it should still be bounded, got {first:.3f}"


def test_the_glycolytic_shift_appears_as_oxygen_runs_out(table):
    """Lactate rises only once oxygen is nearly gone, which is the point.

    A Km of 1 mmHg against 160 mmHg arterial means respiration holds a
    near-maximal rate across the whole perfused range and gives way only in
    the last fraction of a mmHg. So a flat lactate curve over most of the axis
    is the physiology and not a failure to respond; what matters is that the
    transition exists and sits below 1 mmHg.
    """
    glucose = np.full(1, metabolism.GLUCOSE_REFERENCE)
    perfused, _, _ = metabolism.evaluate(
        np.array([metabolism.O2_REFERENCE]), glucose, table)
    lactate_perfused = metabolism.evaluate(
        np.array([metabolism.O2_REFERENCE]), glucose, table)[1][0]
    lactate_anoxic = metabolism.evaluate(np.array([0.0]), glucose, table)[1][0]

    assert lactate_perfused < 0.01, "perfused tissue should not ferment"
    assert lactate_anoxic > 0.5, "anoxic tissue should ferment hard"
    assert perfused[0] == pytest.approx(1.0, abs=1e-6)

    # The transition is below 1 mmHg, in the field's own units.
    one_mmhg = 1.0 / 760.0
    above = metabolism.evaluate(np.array([3 * one_mmhg]), glucose, table)[1][0]
    assert above < 0.01, (
        "at 3 mmHg a cell should still be respiring; if it is fermenting, km "
        "is not where this module thinks it is"
    )


def test_fermentation_defends_atp_before_atp_falls(table):
    """Hypoxia alone is not an energy crisis, and that is correct.

    With glucose available a cell meets its ATP demand by fermenting, so the
    ATP charge holds while lactate climbs. This is the Pasteur effect, and it
    is why hypoxic death in a tumour needs glucose deprivation too -- a model
    in which hypoxia alone starved cells of ATP would be wrong.
    """
    glucose = np.full(1, metabolism.GLUCOSE_REFERENCE)
    low = 0.2 / 760.0            # 0.2 mmHg: deep into the transition
    atp, lactate, feasible = metabolism.evaluate(
        np.array([low]), glucose, table)
    assert feasible[0]
    assert lactate[0] > 0.1, "should be fermenting here"
    assert atp[0] > 0.9, (
        f"ATP charge fell to {atp[0]:.3f} while glucose was plentiful; "
        "fermentation should have covered the demand"
    )


def test_running_out_of_glucose_is_what_cannot_be_survived(table):
    """No carbon source means maintenance cannot be paid, at any oxygen."""
    for o2 in (metabolism.O2_REFERENCE, 0.0):
        atp, lactate, feasible = metabolism.evaluate(
            np.array([o2]), np.array([0.0]), table)
        assert not feasible[0], o2
        assert atp[0] == 0.0
        assert lactate[0] == 0.0


def test_feasibility_is_combined_with_and_not_averaged(table):
    """A cell straddling the feasible boundary is reported infeasible.

    A fractional feasibility is not something a caller can act on, and
    rounding it towards survival is the direction that hides deaths.
    """
    glucose_levels = table.glucose_levels
    # The lowest feasible glucose node, and the infeasible one below it.
    col = int(np.argmax(table.feasible[-1]))
    assert col >= 1, "expected at least one infeasible glucose node"
    between = 0.5 * (glucose_levels[col - 1] + glucose_levels[col])
    _, _, feasible = metabolism.evaluate(
        np.array([metabolism.O2_REFERENCE]), np.array([between]), table)
    assert not feasible[0]


def test_the_caveats_are_carried_with_the_numbers(table):
    """The capacities are not sourced, and the module has to keep saying so."""
    text = metabolism.describe()
    assert "not sourced" in text
    assert "GLUT1" in text or "glucose" in text
    assert len(table.notes) >= 3
