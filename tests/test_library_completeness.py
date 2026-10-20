"""What the library holds, what it refers to, and whether the two meet.

Two gaps were measured and two were closed.

**The cytotoxic effector arm did not exist.** The library held the
apoptotic cascade -- caspase-8, caspase-3, caspase-9, BAX, cytochrome c,
PUMA, MCL-1, XIAP -- and nothing that sets it off. No perforin, no
granzyme, no granulysin, no FasL, no Fas, no TRAIL. The platform's
central mechanism is T-cell and NK killing and the immune module counts
kills on every step, so the chain ran: a killer cell, then nothing, then
a caspase. caspase-8 even carried a note saying it acts at the DISC, and
nothing in the library could form a DISC because neither the receptor
nor its ligand existed.

**Every synonym field was empty.** All 294 entities, since the library
was written. A reference resolves only if the name used is the name
stored, so ``IFNg`` could not reach the ``IFN-gamma`` entity that does
exist.

What is NOT closed, and these tests say so rather than letting it look
finished: 118 of 151 molecules named in the library's own marker and
secretion lists are still not entities. Synonyms cannot conjure them --
derivation doubled what resolves, from 16 to 33, and the rest is
absence, not naming.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library import seed_synonyms as syn                   # noqa: E402
from cognisom.library.seed_cytotoxic_effectors import EFFECTORS     # noqa: E402


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


@pytest.fixture(scope="module")
def rows(store):
    return [tuple(r) for r in store._conn.execute(
        "SELECT entity_type, name, data_json FROM entities")]


def _referenced(rows):
    """Molecules the library's own records name, normalised."""
    out = set()
    for _t, _n, blob in rows:
        props = (json.loads(blob or "{}").get("properties") or {})
        for field in ("surface_markers", "markers", "cytokines_secreting"):
            for value in (props.get(field) or []):
                v = re.sub(r"(hi|lo|low|high|int|dim|bright)$", "",
                           str(value).strip(), flags=re.I)
                out.add(re.sub(r"[+\-]$", "", v).strip())
    return {x for x in out if x}


# ── the effectors ────────────────────────────────────────────────────

def test_the_cytotoxic_effectors_are_in_the_library(store):
    for effector in EFFECTORS:
        entity = store.find_entity_by_name(effector.name)
        assert entity is not None, f"{effector.name} is missing"


def test_each_effector_carries_its_verified_accession(rows):
    by_name = {n: json.loads(b or "{}") for _t, n, b in rows}
    for effector in EFFECTORS:
        props = by_name[effector.name].get("properties") or {}
        assert props.get("uniprot_id") == effector.accession, effector.name
        assert props.get("gene_symbol") == effector.gene
        # The verification record: UniProt's own recommended name, read
        # back before the accession was used.
        assert props.get("uniprot_recommended_name") == effector.uniprot_name


def test_the_kill_chain_reaches_the_caspases_that_were_already_there(rows):
    """The hole this closes. Granzyme B cleaves caspase-3 directly; Fas
    recruits caspase-8. Both targets existed; nothing reached them."""
    by_name = {n: json.loads(b or "{}") for _t, n, b in rows}
    assert "Caspase-3" in by_name and "Caspase-8" in by_name

    def targets(name):
        return {i["target"] for i in (by_name[name].get("interacts_with") or [])}

    assert "Caspase-3" in targets("Granzyme B (GZMB)")
    assert "Caspase-8" in targets("Fas (FAS/CD95)")
    assert "Caspase-8" in targets("TRAIL (TNFSF10/CD253)")
    assert "Fas (FAS/CD95)" in targets("Fas ligand (FASLG/CD178)")
    # perforin opens the way the granzymes go through
    assert "Granzyme B (GZMB)" in targets("Perforin (PRF1)")


def test_both_killing_routes_are_represented(rows):
    """Granule exocytosis and death receptor. A cytotoxic cell uses both
    and they converge on caspase-3."""
    tags = {}
    for _t, n, b in rows:
        d = json.loads(b or "{}")
        if n in {e.name for e in EFFECTORS}:
            tags[n] = set(d.get("tags") or [])
    granule = [n for n, t in tags.items() if "granule_exocytosis" in t]
    receptor = [n for n, t in tags.items() if "death_receptor" in t]
    assert len(granule) >= 4, granule
    assert len(receptor) >= 3, receptor


def test_the_effector_masses_are_the_mature_chains(rows):
    """Sourced from UniProt, and the mature chain rather than the
    precursor: granzyme B is 25.5 kDa once its activation dipeptide is
    gone, against 27.7 for the full translation product."""
    for _t, n, b in rows:
        if n not in {e.name for e in EFFECTORS}:
            continue
        pp = json.loads(b or "{}").get("physics_params") or {}
        mass = pp.get("molecular_weight_kda")
        if n.startswith("Fas ligand"):
            # Refused by the sourcing module, correctly: four declared
            # chains and which one the entity represents is a modelling
            # decision, not something to guess.
            continue
        assert mass is not None, f"{n} has no mass"
        assert 5 < mass < 200, (n, mass)


# ── synonyms ─────────────────────────────────────────────────────────

def test_entities_carry_synonyms_now(rows):
    with_syn = [n for _t, n, b in rows
                if json.loads(b or "{}").get("synonyms")]
    assert len(with_syn) > 150, f"only {len(with_syn)} entities have synonyms"


def test_a_synonym_resolves_to_exactly_one_entity(store):
    """A form two entities answer to is not a synonym: it cannot
    resolve, and whichever is indexed first silently becomes the
    answer."""
    index = syn.resolve_index(store)
    assert index
    # the known-good cases
    for form, want in (("IFNg", "IFN-gamma"), ("CD62L", "L-selectin (CD62L)"),
                       ("GZMB", "Granzyme B (GZMB)"),
                       ("P10144", "Granzyme B (GZMB)")):
        assert index.get(form.lower()) == want, (form, index.get(form.lower()))


def test_ambiguous_forms_are_dropped_not_guessed(store, rows):
    """Three kinds occur and all three are real: a gene and its mutations
    (AR, AR_T878A), a shared subunit (CD18 is in LFA-1 and MAC-1), and a
    locus and its alleles (HLA-A)."""
    ambiguous = syn.ambiguous_forms([(n, b) for _t, n, b in rows])
    assert ambiguous, "nothing was ambiguous, which is implausible here"
    assert "cd18" in ambiguous
    assert len(ambiguous["cd18"]) == 2

    # and none of them was stored as a synonym of either claimant
    for _t, name, blob in rows:
        for stored in (json.loads(blob or "{}").get("synonyms") or []):
            assert stored.strip().lower() not in ambiguous, (name, stored)


def test_greek_letters_are_substituted_not_truncated():
    """Cutting at "kappa" turns NF-kappaB into NF-k, which names
    nothing: the B is part of the molecule."""
    forms = set(syn.derive("NF-kappaB (p65/RELA)"))
    assert "NF-kB" in forms and "NF-κB" in forms
    assert "NF-k" not in forms and "NFk" not in forms
    assert "TNFa" in set(syn.derive("TNF-alpha"))


def test_a_parenthetical_that_is_prose_is_not_made_into_a_molecule():
    """"IgM (pentameric)" does not have a molecule called pentameric."""
    assert "pentameric" not in set(syn.derive("IgM (pentameric)"))
    assert "IgM" in set(syn.derive("IgM (pentameric)"))


# ── what is still missing, stated rather than implied ────────────────

def test_most_referenced_molecules_are_still_not_entities(store, rows):
    """This is the open gap and the test exists to keep it visible. If
    it starts failing because the number fell, that is the work landing
    -- update the bound and say what was added."""
    index = syn.resolve_index(store)
    referenced = _referenced(rows)
    missing = {v for v in referenced if v.lower() not in index}
    assert len(referenced) > 140
    assert 80 < len(missing) <= 118, (
        f"{len(missing)} of {len(referenced)} referenced molecules are not "
        "entities; if this fell, the entity set grew -- raise the bound")


def test_the_biggest_absent_classes_are_the_expected_ones(store, rows):
    index = syn.resolve_index(store)
    missing = {v for v in _referenced(rows) if v.lower() not in index}
    cd = {m for m in missing if m.startswith("CD")}
    chemokine = {m for m in missing if m.startswith(("CCL", "CXCL", "CCR", "CXCR"))}
    factors = {"FOXP3", "T-bet", "GATA3", "RORgt", "BCL6"} & missing
    assert len(cd) > 30, len(cd)
    assert len(chemokine) > 5, chemokine
    # These are the lineage definers the renderer uses to tell a Treg
    # from a Th1. None of them is an entity.
    assert len(factors) >= 4, factors


# ── the effectors reach the simulation ───────────────────────────────

def test_a_cytotoxic_cell_is_loaded_from_the_library():
    """Entities in a library nothing reads are vocabulary nobody speaks.
    An immune cell takes its granule reserve from the entity at
    creation, so the sourced count is load-bearing rather than stored."""
    import numpy as np
    from cognisom.modules.immune_module import ImmuneModule

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        module = ImmuneModule({"n_t_cells": 1, "n_nk_cells": 1,
                               "n_macrophages": 1})
        module.initialize()
    loaded = {c.cell_type: getattr(c, "lytic_granules", None)
              for c in module.immune_cells.values()}
    assert loaded.get("T_cell") == pytest.approx(63)
    assert loaded.get("NK_cell") == pytest.approx(63)
    # None, not zero: the library has no granule count for a macrophage,
    # and zero would say it had already degranulated.
    assert loaded.get("macrophage") is None


def test_the_nk_cell_duplicate_does_not_lose_its_granules():
    """"NK cell" exists as both a cell_type and an immune_cell. Seeding
    by NAME visited one record twice, so the immune_cell copy -- the one
    LINEAGE_ENTITY points the simulation at -- got nothing, and the NK
    cell of all cells ended up unarmed."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        store = EntityStore()
    from cognisom.library.parameters import get_parameter

    rows = [tuple(r) for r in store._conn.execute(
        "SELECT entity_id, entity_type FROM entities WHERE name = 'NK cell'")]
    assert len(rows) == 2, f"the duplicate is gone; this test is stale ({rows})"
    for entity_id, _etype in rows:
        entity = store.get_entity(entity_id)
        pv = get_parameter(entity, "lytic_granule_count")
        assert pv is not None, f"{entity_id} has no granule count"
        assert pv.value == pytest.approx(63)


def test_recording_the_kill_does_not_change_the_kill():
    """The granule draw has its own RNG stream. Drawing from the global
    one consumed numbers the rest of the simulation was going to use, so
    every draw after the first kill shifted: a pinned engagement count
    went 6 -> 2 and a hypoxic death count 60 -> 58, with nothing about
    hypoxia touched."""
    import numpy as np
    from cognisom.modules.immune_module import ImmuneModule

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        module = ImmuneModule({"n_t_cells": 1, "n_nk_cells": 0,
                               "n_macrophages": 0})
        module.initialize()
    assert hasattr(module, "_granule_rng")

    np.random.seed(1234)
    before = np.random.random(5)
    cell = next(iter(module.immune_cells.values()))
    cell.lytic_granules = 63.0

    class _Target:
        cell_id = 99
        position = np.zeros(3, dtype=np.float32)

    np.random.seed(1234)
    module._kill_target(cell, _Target())
    after = np.random.random(5)
    assert np.allclose(before, after), (
        "the kill consumed from the global stream")
    assert cell.lytic_granules < 63.0, "nothing was released"
    assert cell.lytic_granules >= 59.0, "the cell emptied itself"
