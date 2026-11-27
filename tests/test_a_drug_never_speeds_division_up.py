"""A drug mapped as a slowdown must not make a tumour divide faster.

The drug mappings carry a scripted cancer division time -- 18 h for an
anti-androgen -- written against the compressed clock, where an
untreated tumour divides every 8-12 h. On the measured clock the
untreated tumour divides every 1,111 h, and the same 18 h applied
verbatim made the TREATED tumour divide sixty times faster than the
untreated one. A wild-type-AR patient on enzalutamide went from 1,499
to 15,693 cancer cells in four simulated days.

It went unseen because the shipped demo patient carries AR F877L, and
the agonist path does not carry the scripted division time at all. The
first run of a patient WITHOUT that variant on the measured clock, for
longer than a week, was the serum PSA outcome test on 2026-11-27.

These are construction-time tests: the defect is in which number
reaches the cellular module, so no simulation has to run to see it.
"""

from __future__ import annotations

import pytest

from cognisom.dashboard.engine_runner import EngineRunner
from cognisom.modules.cellular_module import MEASURED_DIVISION_HOURS


#: CellularModule's default when a preset names none.
MODULE_DEFAULT_HOURS = 12.0


def test_the_module_default_this_file_assumes_is_the_real_one():
    from cognisom.modules.cellular_module import CellularModule
    assert CellularModule({}).division_time_cancer == MODULE_DEFAULT_HOURS


def _division_time(clock, drugs):
    runner = EngineRunner(dt=0.05, duration=1.0, scenario="Baseline",
                          tissue="ring", drugs=drugs, clock=clock)
    # The compressed Baseline preset sets no division time, so the
    # cellular module's own default is what runs.
    return runner, float(runner.module_configs["cellular"].get(
        "division_time_cancer", MODULE_DEFAULT_HOURS))


def test_on_the_measured_clock_enzalutamide_leaves_the_clock_alone():
    _, untreated = _division_time("measured", None)
    runner, treated = _division_time("measured", ("Enzalutamide",))
    assert untreated == MEASURED_DIVISION_HOURS["cancer"]
    assert treated >= untreated, (
        f"a treated tumour dividing every {treated} h against an "
        f"untreated {untreated} h is the drug speeding the cancer up")
    # and the run says what it declined to apply
    assert any("NOT applied" in n and "division_time_cancer" in n
               for n in runner.treatment.notes), runner.treatment.notes


def test_the_mapping_really_does_carry_a_shorter_number():
    """NON-VACUITY: if the mapping stopped carrying a division time the
    test above would pass with nothing to guard against."""
    from cognisom.workflow.treatment import resolve_treatment
    scripted = resolve_treatment(("Enzalutamide",)).module_configs[
        "cellular"]["division_time_cancer"]
    assert float(scripted) < MEASURED_DIVISION_HOURS["cancer"]


def test_on_the_compressed_clock_the_scripted_slowdown_still_applies():
    """Nothing pinned on the compressed clock may move: there the
    scripted number IS a slowdown and is applied as before."""
    _, untreated = _division_time("compressed", None)
    runner, treated = _division_time("compressed", ("Enzalutamide",))
    assert treated > untreated
    assert treated == pytest.approx(18.0)
    assert not any("NOT applied" in n for n in runner.treatment.notes)


def test_no_mapping_shortens_division_on_the_measured_clock():
    """The rule is general, so check it for every mapping the bridge
    carries a division time in, not only the one that was caught.

    Measured clock only, deliberately. On the compressed clock some
    scenario mappings set 8-10 h on purpose (an aggressive tumour, not
    a drug), the preset holds no number for the guard to compare with,
    and every pinned result there depends on those being applied."""
    clock = "measured"
    from cognisom.workflow import parameter_bridge as pb
    mappings = [v for v in vars(pb).values() if isinstance(v, dict)
                and any(isinstance(x, dict) and "cellular" in x
                        for x in v.values())]
    assert mappings, "found no drug-class mapping to check"
    runner, untreated = _division_time(clock, None)
    checked = 0
    for mapping in mappings:
        for params in mapping.values():
            cellular = (params or {}).get("cellular") or {}
            if "division_time_cancer" not in cellular:
                continue
            runner.treatment = type("T", (), {"notes": []})()
            kept = runner._drug_never_speeds_division("cellular", cellular)
            assert float(kept.get("division_time_cancer", untreated)
                         ) >= untreated
            checked += 1
    assert checked >= 3, checked
