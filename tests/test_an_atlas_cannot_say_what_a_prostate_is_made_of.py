"""The rollup behind the decision not to source DEFAULT_COMPOSITION.

The measurement itself needs the census and the network, so it is not
run here. What is tested is the logic the measurement rests on -- the
ontology rollup and the spread rule -- because that is where a quiet
change would turn "not usable" into a confident wrong correction.

The finding: four CELLxGENE datasets of normal prostate report luminal
cells at anywhere from 7.3% to 89.0%, and basal from 11.0% to 76.2%,
while the worst disagreement with the config is 2.5x. Composition in
dissociated single-cell data is a property of the protocol.
"""

from __future__ import annotations

import math

import pytest

from cognisom.library.tissue_composition import (
    ARCHETYPE_ANCHORS, Composition, UNASSIGNABLE, USABLE_SPREAD, Verdict,
    assign_archetype, spread_of)

# Ancestors as EBI OLS4 returns them, trimmed to the CL terms that
# matter. Recorded rather than fetched so the test does not need a
# network, and so the identical-ancestors fact below is pinned.
ANCESTRY = {
    # luminal and basal prostate epithelium -- note these are EQUAL
    "CL:0002340": ["CL:0000000", "CL:0000066", "CL:0000255", "CL:0002076",
                   "CL:0002231"],
    "CL:0002341": ["CL:0000000", "CL:0000066", "CL:0000255", "CL:0002076",
                   "CL:0002231"],
    # basal epithelial cell of prostatic duct, below basal
    "CL:0002236": ["CL:0000000", "CL:0000066", "CL:0000068", "CL:0000255",
                   "CL:0002076", "CL:0002231", "CL:0002341"],
    # the shared parent, which cannot be split
    "CL:0002231": ["CL:0000000", "CL:0000066", "CL:0000255", "CL:0002076"],
    "CL:0000625": ["CL:0000000", "CL:0000084", "CL:0000219", "CL:0000738"],
    "CL:0002620": ["CL:0000000", "CL:0000057", "CL:0000499"],
    "CL:1000487": ["CL:0000000", "CL:0000192", "CL:0000183"],
    "CL:0000183": ["CL:0000000"],
    "CL:0000034": ["CL:0000000"],
}


def test_luminal_and_basal_prostate_epithelium_are_graph_twins():
    """The fact that forced term-specific anchors.

    If CL ever does subordinate one to the other, this fails and the
    anchors can be simplified -- which is the point of pinning it.
    """
    assert set(ANCESTRY["CL:0002340"]) == set(ANCESTRY["CL:0002341"])
    assert "CL:0000646" not in ANCESTRY["CL:0002341"], (
        "basal prostate epithelium is not filed under 'basal cell'")


def test_the_two_are_still_told_apart():
    assert assign_archetype("CL:0002340", ANCESTRY) == "luminal_secretory"
    assert assign_archetype("CL:0002341", ANCESTRY) == "basal"


def test_a_descendant_rolls_up_without_being_named():
    """Graph ancestry still does the general work.

    `basal epithelial cell of prostatic duct` is not an anchor, and
    `skin fibroblast` is a real label on 1,139 prostate cells.
    """
    assert assign_archetype("CL:0002236", ANCESTRY) == "basal"
    assert assign_archetype("CL:0002620", ANCESTRY) == "fibroblast"


def test_a_parent_term_is_not_claimed_by_its_child():
    """`epithelial cell of prostate` is above both, so it is ambiguous.

    Assigning it to either one would move the very fractions the
    measurement exists to check -- it is 5.3% of the cells.
    """
    assert assign_archetype("CL:0002231", ANCESTRY) is None
    # contractile cell is the PARENT of smooth muscle, not a descendant
    assert assign_archetype("CL:0000183", ANCESTRY) is None


def test_an_unknown_term_is_unassigned_rather_than_guessed():
    assert assign_archetype("CL:0000034", ANCESTRY) is None
    assert assign_archetype("CL:9999999", ANCESTRY) is None


def test_every_unassignable_term_carries_its_reason():
    for curie, why in UNASSIGNABLE.items():
        assert curie.startswith("CL:")
        assert len(why) > 20, f"{curie} has no explanation"


def test_every_anchor_belongs_to_one_archetype():
    seen = {}
    for archetype, ids in ARCHETYPE_ANCHORS.items():
        for curie in ids:
            assert curie not in seen, (
                f"{curie} claimed by {seen.get(curie)} and {archetype}")
            seen[curie] = archetype


# ── the spread rule, which is what makes the verdict falsifiable ──────


def _verdict(*fraction_sets):
    v = Verdict()
    v.per_dataset = [Composition(label=f"d{i}", cells=5000, fractions=f)
                     for i, f in enumerate(fraction_sets)]
    v.spread = spread_of(v.per_dataset)
    return v


def test_a_zero_in_one_dataset_is_infinite_disagreement():
    """A dataset finding no macrophages does not agree with one at 5%."""
    v = _verdict({"macrophage": 0.05}, {"macrophage": 0.0})
    assert math.isinf(v.spread["macrophage"])
    assert not v.usable


def test_the_threshold_is_stricter_than_the_error_it_judges():
    """A rule looser than the discrepancy would wave anything through."""
    assert USABLE_SPREAD < 2.5


def test_sources_that_agree_are_usable():
    """The rule has to be able to pass, or it asserts nothing."""
    v = _verdict({"basal": 0.20, "t_cell": 0.05},
                 {"basal": 0.25, "t_cell": 0.06})
    assert v.usable, v.spread


def test_the_real_prostate_numbers_fail_the_rule():
    """The measured spread, as four datasets actually reported it."""
    v = _verdict(
        {"luminal_secretory": 0.428, "basal": 0.281},
        {"luminal_secretory": 0.520, "basal": 0.170},
        {"luminal_secretory": 0.073, "basal": 0.762},
        {"luminal_secretory": 0.890, "basal": 0.110},
    )
    assert v.spread["luminal_secretory"] == pytest.approx(12.2, abs=0.1)
    assert v.spread["basal"] == pytest.approx(6.9, abs=0.1)
    assert not v.usable
    # The whole argument in one line: the evidence disagrees with
    # itself by more than the config disagrees with the evidence, so
    # moving the config toward it would be fitting noise.
    worst_disagreement_with_config = 2.5
    assert min(v.spread.values()) > worst_disagreement_with_config


def test_the_config_still_sums_to_one():
    """Whatever else is true, it is a composition."""
    from cognisom.core.tissue_config import DEFAULT_COMPOSITION

    assert sum(DEFAULT_COMPOSITION.values()) == pytest.approx(1.0)


def test_the_config_says_it_is_not_sourced_from_an_atlas():
    """The comment is the only thing stopping the next correction."""
    import pathlib

    import cognisom.core.tissue_config as tc

    source = pathlib.Path(tc.__file__).read_text()
    head = source[:source.index("DEFAULT_COMPOSITION")]
    assert "DO NOT SOURCE THEM FROM A SINGLE-CELL ATLAS" in head
    assert "tissue_composition" in head
