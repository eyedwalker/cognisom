"""A T cell that recognises its antigen divides. Until now none ever did.

The gap this closes
-------------------
``ImmuneModule`` recruited T cells and let them kill, but no T cell ever
divided. That leaves out the mechanism most of adaptive immunity runs on,
and it matters for the one treatment the engine already models: PD-1
blockade works largely through a CD28-dependent burst of CD8 proliferation
(Kamphorst 2017, PMID 28280249 -- "the CD28/B7 costimulatory pathway is
essential for ... CD8 proliferation after PD-1 blockade"; the engine
already cites that paper for the CD28 relief term in ``tcell_kill``). With
no proliferation, blockade could only ever change how hard existing cells
kill, never how many there are.

Division is a PROGRAM, not a response to standing antigen
---------------------------------------------------------
Two 2001 papers settled the shape of this, and the shape is what decides
the code:

* Kaech & Ahmed, Nat Immunol 2:415-422, **PMID 11323695**: once the
  parental naive CD8 T cell has been activated the process "could not be
  interrupted and the daughter cells continued to divide and differentiate
  in the absence of further antigenic stimulation", and every recruited
  cell "divided at least seven times after initial antigen encounter".
* van Stipdonk, Lemmens & Schoenberger, Nat Immunol 2:423-429,
  **PMID 11323696**: naive CTLs require only "a single brief period of
  antigenic stimulation for clonal expansion and differentiation".

So the trigger is the FIRST recognition and nothing after it. A cell that
engages once and then loses its target keeps dividing; standing antigen is
not required and is not read here. That is why ``divisions_due`` is a pure
function of a clock and takes no antigen argument -- had it been written as
"divide while a target is in contact", the engine would have produced
expansion only where killing was already happening, which is the opposite
of what these two papers measured.

How fast
--------
Yoon, Kim & Braciale, PLoS ONE 5(11):e15423, **PMID 21079741**, measured
cell cycle times of CD8 T cells responding *in vivo* (murine, draining
lymph node, CFSE dilution plus DNA content), and the headline is that the
rate is set by the stimulus rather than fixed:

    stimulus                                        initial cycle time
    influenza A infection, intranasal                   ~2 h
    influenza-infected dendritic cells, i.v.            ~3-4 h
    vaccinia infection                                  ~5-6 h
    synthetic peptide + adjuvant (any adjuvant/route)   ~6 h
    low-avidity altered peptide ligand                  ~3-4 h
    (prior literature, activated lymphocytes)           6-8 h

and it slows "after the cells had undergone 5 or more divisions".

**This engine's default is the slowest of those, 6 h, and it is an
ESTIMATE rather than a sourced value.** Every number above is a mouse
responding to an acute infection or a vaccine in a lymph node. This engine
simulates established tumour tissue, where expansion is slower, and it has
no lymph node and no priming at all (see the platform roadmap): its T
cells arrive as effectors. An intratumoural per-cell cycle time does not
exist in the literature -- the usual intratumoural measurement is a Ki-67+
FRACTION, which is a growing fraction and not a rate, and converting one
to the other needs a cycle length nobody measured. Taking the fastest
figure, or any figure, as if it applied here would be the same mistake.

Deliberately NOT wired
----------------------
* **Suppression does not slow division.** Chen et al. 2005
  (PMID 15623559), which this engine already uses for the Treg to perforin
  route, measured recognition and expansion LEFT UNCHANGED while
  cytotoxicity was suppressed. ``treg_suppression`` therefore acts on
  perforin, and expansion must not read it. Wiring Treg suppression here
  would contradict the paper the suppression term came from.
* **Exhaustion does stop it.** An exhausted clone is a fixed,
  non-rescuable state (Philip 2017, PMID 28514453, already the basis of
  ``tcell_exhaustion_onset_hours``), so it does not expand.

This module holds the numbers and the arithmetic only. It decides nothing
about who is in contact with whom; ``ImmuneModule`` applies it, behind
``tcell_expansion_enabled``, which is **off by default** so that no pinned
outcome moves until the effect has been measured and the default is a
deliberate decision.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

# The same "a number, and what stands behind it" record the kill path
# uses. Its docstring says "on the kill path"; the shape is generic and a
# second identical dataclass would be a copy that can drift.
from cognisom.engine.py.immune.kill_calibration import KillParameter

_YOON = "Yoon 2010, PMID 21079741"
_KAECH = "Kaech & Ahmed 2001, PMID 11323695"

#: Hours between divisions of a T cell running its proliferative program.
#:
#: 6 h is the SLOWEST initial cycle time Yoon measured (synthetic peptide
#: with adjuvant, uniform across adjuvants and routes) and matches the
#: 6-8 h classical figure for activated lymphocytes. The range spans the
#: fastest stimulus (influenza, 2 h) to that classical upper bound.
#: ESTIMATE, not sourced: every measurement behind it is a mouse lymph
#: node in acute infection, and this engine runs established tumour
#: tissue with no priming compartment at all.
DIVISION_HOURS = KillParameter(
    6.0, 2.0, 8.0, 'ESTIMATE',
    f"{_YOON}: in vivo CD8 initial cycle time is stimulus-dependent, "
    "~2 h (influenza) to ~6 h (peptide+adjuvant), 6-8 h classical; the "
    "slowest is taken because no intratumoural per-cell rate exists")

#: Divisions one activated clone completes before its program ends.
#:
#: Kaech & Ahmed found every recruited CD8 T cell divided at least seven
#: times after its initial antigen encounter, and that the program runs
#: without further antigen. Seven is therefore a FLOOR in their system,
#: used here as the cap: this engine's T cells are already effectors
#: rather than naive precursors, so the full naive program would overstate
#: what an intratumoural cell has left in it.
PROGRAM_DIVISIONS = KillParameter(
    7, 7, 19, 'ESTIMATE',
    f"{_KAECH}: all recruited cells divided >= 7 times after initial "
    "antigen encounter, antigen-independently; >= 7 is a floor there and "
    "is used as a cap here because these cells start as effectors")

#: Hours after its first recognition that a cell's first division lands.
#: The same clock as the interval: Yoon's figures are the interval between
#: divisions, and the first one is counted from activation.
FIRST_DIVISION_HOURS = DIVISION_HOURS


def divisions_due(first_engaged_h: Optional[float],
                  divisions_done: int,
                  now_h: float,
                  cycle_h: float = DIVISION_HOURS.value,
                  program_divisions: int = int(PROGRAM_DIVISIONS.value),
                  ) -> int:
    """How many divisions this cell owes, right now.

    A pure function of a clock, with no antigen argument, because Kaech &
    Ahmed and van Stipdonk both measured that the program continues
    without further antigenic stimulation.

    ``first_engaged_h`` is when the cell first recognised its antigen, or
    None if it never has -- an unengaged T cell never divides, which is
    what makes this proliferation ANTIGEN-DRIVEN rather than a background
    birth rate.
    """
    if first_engaged_h is None or cycle_h <= 0.0 or program_divisions <= 0:
        return 0
    elapsed = now_h - float(first_engaged_h)
    if elapsed < cycle_h:
        return 0
    earned = min(int(math.floor(elapsed / cycle_h)), int(program_divisions))
    return max(0, earned - int(divisions_done))


def fold_expansion(divisions: int) -> float:
    """What ``divisions`` rounds of division multiply one cell into."""
    return float(2 ** max(0, int(divisions)))


def expansion_report(enabled: bool,
                     cycle_h: float = DIVISION_HOURS.value,
                     program_divisions: int = int(PROGRAM_DIVISIONS.value),
                     divided: int = 0,
                     founders: int = 0,
                     ) -> Dict[str, object]:
    """What a run can say about this mechanism, including when it is off."""
    return {
        'enabled': bool(enabled),
        'division_hours': {**DIVISION_HOURS.as_dict(), 'in_use': cycle_h},
        'program_divisions': {**PROGRAM_DIVISIONS.as_dict(),
                              'in_use': program_divisions},
        'divisions': int(divided),
        'founders': int(founders),
        'suppression_applies': False,
        'suppression_reason': (
            'Chen 2005 (PMID 15623559) measured expansion unchanged while '
            'cytotoxicity was suppressed; Treg acts on perforin instead'),
        'exhaustion_applies': True,
        'caveat': (
            'Every measurement behind these numbers is a mouse lymph node '
            'in acute infection. This engine has no priming compartment, '
            'so its T cells start as effectors; no intratumoural per-cell '
            'cycle time exists in the literature.'),
    }
