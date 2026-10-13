"""The interaction layer must not draw things that did not happen there, then.

Every test here guards a specific way a render can lie:

* a marker drawn at a cell's *current* position for an event that
  happened earlier -- the recorder keeps the coordinate the emitter put
  in the payload and never refreshes it;
* a position invented for an event that carried none -- ``hypoxia_detected``
  reports a count and a severity and no coordinates;
* an attacker's end of a kill guessed from wherever it has drifted to --
  drawn only when it was resolved inside the event callback;
* a marker that is *invisible on the frame the event happened*, which is
  the one frame the layer exists for;
* a marker that bleeds backwards across a whole run because USD
  interpolates sparse time samples linearly -- measured, a division
  burst belonging on frame 181 was at 45% size on frame 149;
* an oxygen picture that shows a smooth field where the solver produced
  a uniform one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cognisom.core.event_bus import EventBus, EventTypes
from cognisom.visualization.interaction_recorder import (
    DEFAULT_EVENT_KINDS,
    InteractionRecorder,
    KIND_ACTIVATION,
    KIND_DEATH,
    KIND_DIVISION,
    KIND_HYPOXIA,
    KIND_KILL,
    VisualEvent,
)


class FakeClock:
    """Stands in for the engine's time and step counters."""

    def __init__(self):
        self.time = 0.0
        self.step = 0

    def advance(self, dt=0.05):
        self.time += dt
        self.step += 1


def _recorder(clock, **kw):
    return InteractionRecorder(
        time_fn=lambda: clock.time, step_fn=lambda: clock.step, **kw)


# ── The recorder ─────────────────────────────────────────────────────

def test_position_is_the_one_the_event_carried_not_the_cell_s_later_one():
    """A kill marker belongs at the kill site, forever.

    The payload position is a list the emitter built from the cell's
    coordinates in the step the event fired. Mutating the cell
    afterwards -- which is what a moving cell does every step -- must not
    move the marker.
    """
    bus = EventBus()
    clock = FakeClock()
    rec = _recorder(clock).attach(bus)

    cell_position = [10.0, 20.0, 30.0]
    bus.emit(EventTypes.CANCER_KILLED, {
        "cell_id": 7, "killer_id": 3, "killer_type": "NK_cell",
        "position": cell_position})
    bus.process_events()

    # The cell carries on living its life somewhere else entirely.
    cell_position[0] = 999.0
    for _ in range(40):
        clock.advance()

    assert len(rec.events) == 1
    assert rec.events[0].position == (10.0, 20.0, 30.0)
    assert rec.events[0].provenance == "event_payload"


def test_time_and_step_come_from_the_moment_of_delivery():
    bus = EventBus()
    clock = FakeClock()
    rec = _recorder(clock).attach(bus)

    for _ in range(4):
        clock.advance(0.25)
    bus.emit(EventTypes.CELL_DIVIDED, {
        "cell_id": 1, "daughter_id": 2, "cell_type": "cancer",
        "position": [1.0, 2.0, 3.0]})
    bus.process_events()
    for _ in range(10):
        clock.advance(0.25)

    assert rec.events[0].time == pytest.approx(1.0)
    assert rec.events[0].step == 4


def test_an_event_with_no_position_is_kept_but_never_placed():
    """HYPOXIA_DETECTED carries a count and a severity, and no coordinates."""
    bus = EventBus()
    rec = _recorder(FakeClock()).attach(bus)

    bus.emit(EventTypes.HYPOXIA_DETECTED,
             {"n_hypoxic_cells": 3, "severity": "moderate"})
    bus.process_events()

    event = rec.events[0]
    assert event.kind == KIND_HYPOXIA
    assert event.position is None
    assert event.provenance == "none"
    assert rec.global_events == [event]
    assert rec.positioned_events == []


def test_no_attacker_position_without_a_lookup():
    """The kill payload names the killer but does not locate it."""
    bus = EventBus()
    rec = _recorder(FakeClock()).attach(bus)
    bus.emit(EventTypes.CANCER_KILLED, {
        "cell_id": 7, "killer_id": 3, "killer_type": "T_cell",
        "position": [1.0, 1.0, 1.0]})
    bus.process_events()

    assert rec.events[0].partner_position is None
    assert rec.events[0].partner_provenance == "none"


def test_attacker_position_is_resolved_inside_the_callback():
    """With a lookup, the attacker's coordinates are read at emission.

    The lookup here returns whatever the attacker's position is *at the
    moment it is called*. Calling it later would give a different,
    wrong answer -- which is exactly what the test moves it to prove.
    """
    bus = EventBus()
    attacker = {"position": [4.0, 5.0, 6.0]}
    rec = _recorder(
        FakeClock(),
        position_lookup=lambda cid: attacker["position"] if cid == 3 else None,
    ).attach(bus)

    bus.emit(EventTypes.CANCER_KILLED, {
        "cell_id": 7, "killer_id": 3, "killer_type": "T_cell",
        "position": [1.0, 1.0, 1.0]})
    bus.process_events()
    attacker["position"] = [900.0, 900.0, 900.0]

    assert rec.events[0].partner_position == (4.0, 5.0, 6.0)
    assert rec.events[0].partner_provenance == "live_at_emission"


def test_a_failing_lookup_leaves_the_attacker_unplaced():
    bus = EventBus()

    def boom(_cid):
        raise KeyError("gone")

    rec = _recorder(FakeClock(), position_lookup=boom).attach(bus)
    bus.emit(EventTypes.CANCER_KILLED, {
        "cell_id": 7, "killer_id": 3, "position": [1.0, 1.0, 1.0]})
    bus.process_events()

    assert rec.events[0].partner_position is None
    assert rec.events[0].partner_provenance == "none"


def test_a_malformed_position_is_refused_rather_than_padded():
    bus = EventBus()
    rec = _recorder(FakeClock()).attach(bus)
    bus.emit(EventTypes.CELL_DIED,
             {"cell_id": 1, "cause": "apoptosis", "position": [1.0, 2.0]})
    bus.process_events()
    assert rec.events[0].position is None


def test_tcell_kills_and_innate_kills_are_the_same_picture():
    assert DEFAULT_EVENT_KINDS["cancer_killed"] == KIND_KILL
    assert DEFAULT_EVENT_KINDS["cell_killed_by_tcell"] == KIND_KILL
    assert DEFAULT_EVENT_KINDS["immune_activated"] == KIND_ACTIVATION
    assert DEFAULT_EVENT_KINDS["cell_divided"] == KIND_DIVISION
    assert DEFAULT_EVENT_KINDS["cell_died"] == KIND_DEATH


def test_high_traffic_events_are_not_recorded():
    """gene_expressed fires 1662 times in six hours and has no position."""
    assert "gene_expressed" not in DEFAULT_EVENT_KINDS
    assert "circadian_peak" not in DEFAULT_EVENT_KINDS


def test_engagements_report_how_long_the_attacker_was_engaged():
    """Measured: one engagement in a 48 h run ran 11.95 h -> 30.00 h.

    That is why the layer does not draw a line between the activation
    position and the kill position: they are eighteen hours apart.
    """
    bus = EventBus()
    clock = FakeClock()
    rec = _recorder(clock).attach(bus)

    clock.time = 11.95
    bus.emit(EventTypes.IMMUNE_ACTIVATED, {
        "immune_id": 44, "immune_type": "NK_cell", "target_id": 104,
        "position": [1.0, 1.0, 1.0]})
    bus.process_events()
    clock.time = 30.0
    bus.emit(EventTypes.CANCER_KILLED, {
        "cell_id": 104, "killer_id": 44, "killer_type": "NK_cell",
        "position": [2.0, 2.0, 2.0]})
    bus.process_events()

    engagements = rec.engagements()
    assert len(engagements) == 1
    assert engagements[0].duration_hours == pytest.approx(18.05)
    assert engagements[0].activation_position == (1.0, 1.0, 1.0)
    assert engagements[0].kill_position == (2.0, 2.0, 2.0)


def test_detach_stops_recording():
    bus = EventBus()
    rec = _recorder(FakeClock()).attach(bus)
    rec.detach()
    bus.emit(EventTypes.CELL_DIVIDED,
             {"cell_id": 1, "position": [0.0, 0.0, 0.0]})
    bus.process_events()
    assert rec.events == []


def test_attaching_twice_is_refused():
    bus = EventBus()
    rec = _recorder(FakeClock()).attach(bus)
    with pytest.raises(RuntimeError):
        rec.attach(bus)


def test_round_trip_through_dicts_preserves_provenance():
    bus = EventBus()
    rec = _recorder(
        FakeClock(), position_lookup=lambda cid: [4.0, 5.0, 6.0]).attach(bus)
    bus.emit(EventTypes.CANCER_KILLED, {
        "cell_id": 7, "killer_id": 3, "killer_type": "NK_cell",
        "position": [1.0, 2.0, 3.0]})
    bus.emit(EventTypes.HYPOXIA_DETECTED,
             {"n_hypoxic_cells": 2, "severity": "moderate"})
    bus.process_events()

    restored = InteractionRecorder.events_from_dicts(rec.to_dicts())
    assert [e.kind for e in restored] == [e.kind for e in rec.events]
    assert restored[0].position == (1.0, 2.0, 3.0)
    assert restored[0].partner_provenance == "live_at_emission"
    assert restored[1].position is None
    assert restored[1].provenance == "none"


def test_captured_fields_are_copies():
    """The engine mutates its field arrays in place."""
    rec = _recorder(FakeClock())
    arr = np.zeros((2, 2, 2), dtype=np.float32)
    rec.capture_field("oxygen", arr, 0.0, resolution=10.0)
    arr[:] = 0.21
    rec.capture_field("oxygen", arr, 1.0, resolution=10.0)

    frames = rec.field_frames["oxygen"]
    assert frames[0][1].max() == pytest.approx(0.0)
    assert frames[1][1].max() == pytest.approx(0.21)


def test_max_events_counts_the_overflow_instead_of_evicting():
    bus = EventBus()
    rec = _recorder(FakeClock(), max_events=2).attach(bus)
    for i in range(5):
        bus.emit(EventTypes.CELL_DIVIDED,
                 {"cell_id": i, "position": [0.0, 0.0, 0.0]})
    bus.process_events()
    assert len(rec.events) == 2
    assert rec.dropped == 3
    assert rec.events[0].cell_id == 0   # the head is what is kept


# ── Timing ───────────────────────────────────────────────────────────

usd_geom = pytest.importorskip(
    "pxr.UsdGeom", reason="OpenUSD (usd-core) not installed")

from pxr import Usd, UsdGeom  # noqa: E402

from cognisom.visualization import interaction_usd as iusd  # noqa: E402
from cognisom.visualization.interaction_usd import (  # noqa: E402
    DEFAULT_STYLES, FrameMap, envelope_scale,
)


def test_frame_map_places_an_event_on_the_first_frame_at_or_after_it():
    fm = FrameMap([0.0, 0.2, 0.4, 0.6])
    assert fm.frame_of(0.0) == 0
    assert fm.frame_of(0.1) == 1      # happened between frame 0 and 1
    assert fm.frame_of(0.2) == 1
    assert fm.frame_of(0.6) == 3


def test_frame_map_drops_an_event_past_the_end_rather_than_clamping():
    """Clamping would pile a run's whole tail onto the final image."""
    fm = FrameMap([0.0, 0.2, 0.4])
    assert fm.frame_of(0.5) is None


def test_frame_map_rejects_a_non_monotonic_timeline():
    with pytest.raises(ValueError):
        FrameMap([0.0, 0.4, 0.2])
    with pytest.raises(ValueError):
        FrameMap([])


@pytest.mark.parametrize("envelope", ["burst", "flash", "collapse"])
def test_a_marker_is_visible_on_the_frame_its_event_happened(envelope):
    """The regression that this guards made every kill invisible.

    The first envelope ramped from zero over the first quarter of the
    marker's life, so at age 0 -- the frame of the event -- the scale was
    exactly 0.0. Rendering frame 151 of a run with a kill on frame 151
    showed no kill.
    """
    assert envelope_scale(envelope, 0, 8) > 0.0


@pytest.mark.parametrize("envelope", ["burst", "flash", "collapse"])
def test_a_marker_is_gone_outside_its_lifetime(envelope):
    assert envelope_scale(envelope, -1, 6) == 0.0
    assert envelope_scale(envelope, 6, 6) == 0.0
    assert envelope_scale(envelope, 99, 6) == 0.0


# ── Authoring ────────────────────────────────────────────────────────

def _stage():
    return Usd.Stage.CreateInMemory()


def _kill(time, position=(10.0, 10.0, 10.0), partner=None):
    return VisualEvent(
        kind=KIND_KILL, event_type="cancer_killed", time=time, step=0,
        position=position, cell_id=1, partner_id=2,
        partner_position=partner,
        partner_provenance="live_at_emission" if partner else "none",
        detail={"killer_type": "NK_cell"})


def _scales_at(stage, path, frame):
    pi = UsdGeom.PointInstancer.Get(stage, path)
    assert pi, f"no PointInstancer at {path}"
    return [float(v[0]) for v in pi.GetScalesAttr().Get(frame)]


def test_a_marker_does_not_bleed_into_frames_between_distant_events():
    """USD interpolates numeric time samples linearly.

    With samples authored only where values change, a marker whose event
    is on frame 40 is already partly grown on frame 5, because the stage
    ramps from the frame-0 sample to the frame-40 one. Measured in a real
    run: 59 division markers on screen 32 frames early.
    """
    stage = _stage()
    fm = FrameMap([i * 0.2 for i in range(60)])
    events = [_kill(0.2 * 40)]
    iusd.author_interaction_markers(stage, events, fm)

    path = "/World/Interactions/Kill"
    assert _scales_at(stage, path, 5) == [0.0]
    assert _scales_at(stage, path, 39) == [0.0]
    assert _scales_at(stage, path, 40)[0] > 0.0


def test_a_marker_turns_off_after_its_lifetime():
    stage = _stage()
    fm = FrameMap([i * 0.2 for i in range(60)])
    style = DEFAULT_STYLES[KIND_KILL]
    iusd.author_interaction_markers(stage, [_kill(0.2 * 10)], fm)
    path = "/World/Interactions/Kill"
    assert _scales_at(stage, path, 10 + style.lifetime_frames) == [0.0]


def test_invisible_ids_and_zero_scale_agree():
    stage = _stage()
    fm = FrameMap([i * 0.2 for i in range(40)])
    iusd.author_interaction_markers(
        stage, [_kill(0.2 * 5), _kill(0.2 * 30)], fm)
    pi = UsdGeom.PointInstancer.Get(stage, "/World/Interactions/Kill")
    for frame in (0, 5, 7, 20, 30, 39):
        scales = [float(v[0]) for v in pi.GetScalesAttr().Get(frame)]
        hidden = set(int(i) for i in pi.GetInvisibleIdsAttr().Get(frame))
        assert hidden == {i for i, s in enumerate(scales) if s <= 0.0}


def test_an_event_with_no_position_is_counted_not_drawn():
    stage = _stage()
    fm = FrameMap([0.0, 0.2, 0.4])
    ghost = VisualEvent(kind=KIND_HYPOXIA, event_type="hypoxia_detected",
                        time=0.2, step=1, position=None, provenance="none")
    report = iusd.author_interaction_markers(stage, [ghost], fm)
    assert report["no_position"] == 1
    assert report["drawn"] == {}


def test_an_immune_kill_death_is_not_drawn_twice():
    """CANCER_KILLED and CELL_DIED fire at the same place and instant."""
    stage = _stage()
    fm = FrameMap([0.0, 0.2, 0.4])
    death = VisualEvent(
        kind=KIND_DEATH, event_type="cell_died", time=0.2, step=1,
        position=(1.0, 1.0, 1.0), detail={"cause": "immune_killed"})
    apoptosis = VisualEvent(
        kind=KIND_DEATH, event_type="cell_died", time=0.2, step=1,
        position=(2.0, 2.0, 2.0), detail={"cause": "apoptosis"})
    report = iusd.author_interaction_markers(stage, [death, apoptosis], fm)
    assert report["suppressed_duplicate_death"] == 1
    assert report["drawn"][KIND_DEATH] == 1


def test_an_event_after_the_last_frame_is_reported_not_clamped():
    stage = _stage()
    fm = FrameMap([0.0, 0.2, 0.4])
    report = iusd.author_interaction_markers(stage, [_kill(9.0)], fm)
    assert report["after_last_frame"] == 1
    assert report["drawn"] == {}


def test_no_bond_without_an_attacker_position():
    stage = _stage()
    fm = FrameMap([i * 0.2 for i in range(20)])
    report = iusd.author_engagement_bonds(stage, [_kill(0.2 * 5)], fm)
    assert report["drawn"] == 0
    assert report["no_partner_position"] == 1
    assert not stage.GetPrimAtPath("/World/Interactions/Engagements")


def test_a_bond_is_drawn_only_while_the_kill_is_fresh():
    stage = _stage()
    fm = FrameMap([i * 0.2 for i in range(20)])
    event = _kill(0.2 * 5, partner=(20.0, 10.0, 10.0))
    report = iusd.author_engagement_bonds(stage, [event], fm, hold_frames=4)
    assert report["drawn"] == 1

    curves = UsdGeom.BasisCurves.Get(stage, "/World/Interactions/Engagements")
    assert list(curves.GetCurveVertexCountsAttr().Get()) == [2]
    points = [tuple(float(c) for c in p) for p in curves.GetPointsAttr().Get()]
    assert points == [(20.0, 10.0, 10.0), (10.0, 10.0, 10.0)]
    assert float(curves.GetWidthsAttr().Get(4)[0]) == 0.0
    assert float(curves.GetWidthsAttr().Get(5)[0]) > 0.0
    assert float(curves.GetWidthsAttr().Get(9)[0]) == 0.0


def test_a_bond_needs_a_live_at_emission_attacker():
    """A coordinate from any other source is refused.

    partner_position set without the provenance to back it means it came
    from somewhere other than the kill instant, and a bond drawn from it
    would claim an engagement somewhere the cells were not.
    """
    stage = _stage()
    fm = FrameMap([i * 0.2 for i in range(20)])
    stale = VisualEvent(
        kind=KIND_KILL, event_type="cancer_killed", time=1.0, step=5,
        position=(10.0, 10.0, 10.0), partner_position=(20.0, 10.0, 10.0),
        partner_provenance="none")
    report = iusd.author_engagement_bonds(stage, [stale], fm)
    assert report["drawn"] == 0


# ── Oxygen ───────────────────────────────────────────────────────────

def test_a_uniform_oxygen_field_is_refused_with_its_measured_range():
    """Measured on this engine: range 2.3e-06 over a 12 h Hypoxia run."""
    stage = _stage()
    fm = FrameMap([0.0, 0.2, 0.4])
    flat = np.full((4, 4, 3), 0.21, dtype=np.float32)
    report = iusd.author_oxygen_slice(
        stage, [(0.0, flat), (0.2, flat)], fm, resolution=10.0)
    assert report["authored"] is False
    assert report["measured_range"] == pytest.approx(0.0)
    assert "uniform" in report["reason"]
    assert not stage.GetPrimAtPath("/World/Fields/OxygenSlice")


def test_a_field_with_structure_is_drawn():
    stage = _stage()
    fm = FrameMap([0.0, 0.2])
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    arr[0, :, :] = 0.21
    report = iusd.author_oxygen_slice(
        stage, [(0.0, arr), (0.2, arr)], fm, resolution=10.0)
    assert report["authored"] is True
    mesh = UsdGeom.Mesh.Get(stage, "/World/Fields/OxygenSlice")
    assert len(mesh.GetPointsAttr().Get()) == 4 * 4


def test_oxygen_colour_breaks_at_the_models_own_threshold():
    anoxic = iusd.oxygen_color(0.0)
    hypoxic = iusd.oxygen_color(0.04)
    perfused = iusd.oxygen_color(0.21)
    assert hypoxic[0] > hypoxic[2]          # red dominates below threshold
    assert perfused[2] > perfused[0]        # blue-green above it
    assert anoxic[0] < hypoxic[0]           # darker the less there is


def test_hypoxic_halos_follow_the_recorded_per_cell_oxygen():
    """A cell gains its halo on the frame its oxygen crossed, not before."""
    stage = _stage()
    fm = FrameMap([0.0, 0.2, 0.4, 0.6])
    positions = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 10.0]])
    frames = [
        (positions, np.array([0.21, 0.21])),
        (positions, np.array([0.21, 0.21])),
        (positions, np.array([0.01, 0.21])),   # cell 0 becomes hypoxic
        (positions, np.array([0.01, 0.02])),   # and then cell 1 too
    ]
    report = iusd.author_cell_oxygen(stage, frames, fm)
    assert report["peak_hypoxic_cells"] == 2
    assert report["frames_with_hypoxia"] == 2

    halos = UsdGeom.PointInstancer.Get(stage, "/World/Fields/HypoxicCells")
    assert len(halos.GetInvisibleIdsAttr().Get(0)) == 2    # none shown
    assert len(halos.GetInvisibleIdsAttr().Get(2)) == 1    # one shown
    assert len(halos.GetInvisibleIdsAttr().Get(3)) == 0    # both shown


def test_no_hypoxia_marker_when_nothing_was_ever_hypoxic():
    stage = _stage()
    fm = FrameMap([0.0, 0.2])
    positions = np.array([[10.0, 10.0, 10.0]])
    frames = [(positions, np.array([0.21]))] * 2
    report = iusd.author_cell_oxygen(stage, frames, fm)
    assert report["halos"] is None
    assert "no cell was ever below" in report["halo_reason"]
    assert not stage.GetPrimAtPath("/World/Fields/HypoxicCells")


def test_oxygen_shadows_carry_one_colour_per_cell():
    stage = _stage()
    fm = FrameMap([0.0, 0.2])
    positions = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 10.0]])
    frames = [(positions, np.array([0.21, 0.01]))] * 2
    iusd.author_cell_oxygen(stage, frames, fm, draw_halos=False)
    pi = UsdGeom.PointInstancer.Get(stage, "/World/Fields/OxygenShadows")
    colors = UsdGeom.PrimvarsAPI(pi.GetPrim()).GetPrimvar(
        "displayColor").Get(0)
    assert len(colors) == 2
    assert colors[0][2] > colors[0][0]      # perfused: blue-green
    assert colors[1][0] > colors[1][2]      # hypoxic: red


def test_oxygen_frames_adapter_keeps_empty_frames_in_place():
    """Dropping an empty frame would shift every later frame's index."""
    class Snap:
        def __init__(self, pos, oxy):
            self.cell_positions = pos
            self.cell_oxygen = oxy

    frames = iusd.oxygen_frames_from_snapshots([
        Snap(None, []),
        Snap(np.array([[1.0, 2.0, 3.0]]), [0.2]),
    ])
    assert len(frames) == 2
    assert len(frames[0][0]) == 0
    assert frames[1][1][0] == pytest.approx(0.2)


# ── Camera and legend ────────────────────────────────────────────────

def test_the_camera_pulls_back_far_enough_to_contain_the_tissue():
    stage = _stage()
    info = iusd.author_event_camera(
        stage, (100.0, 100.0, 50.0), 150.0, focal_length=35.0,
        distance_scale=0.5)          # deliberately much too close
    half_fov = math.atan((info["aperture"] / 2.0) / info["focal_length"])
    assert info["distance"] >= 150.0 / math.sin(half_fov)


def test_the_camera_authors_a_backdrop_behind_everything():
    stage = _stage()
    info = iusd.author_event_camera(stage, (0.0, 0.0, 0.0), 10.0)
    backdrop = UsdGeom.Mesh.Get(stage, info["path"] + "/Backdrop")
    assert backdrop
    zs = [float(p[2]) for p in backdrop.GetPointsAttr().Get()]
    assert max(zs) < -info["distance"]      # behind the subject, in camera space


def test_the_backdrop_and_legend_are_inside_the_clipping_range():
    """Both are camera children, and both can fall outside the frustum.

    The first version set the clipping range from the requested distance
    and only then pulled the camera back to fit the subject, so the far
    plane landed in front of the backdrop and the render came out on a
    transparent background. The near plane has the mirror problem: it
    was tightened to kill z-fighting on the oxygen floor, which would
    have clipped a legend quad still hanging at one unit.
    """
    stage = _stage()
    info = iusd.author_event_camera(
        stage, (0.0, 0.0, 0.0), 150.0, distance_scale=0.5)
    cam = UsdGeom.Camera.Get(stage, info["path"])
    near, far = (float(v) for v in cam.GetClippingRangeAttr().Get())

    backdrop = UsdGeom.Mesh.Get(stage, info["path"] + "/Backdrop")
    depths = [abs(float(p[2])) for p in backdrop.GetPointsAttr().Get()]
    assert near < min(depths)
    assert max(depths) < far
    assert near < info["legend_distance"] < far


def test_the_legend_lists_only_the_kinds_present():
    rows = iusd.legend_entries_for([KIND_KILL, KIND_DIVISION])
    assert [r["label"] for r in rows] == [
        DEFAULT_STYLES[KIND_KILL].label, DEFAULT_STYLES[KIND_DIVISION].label]
    assert iusd.legend_entries_for([]) == []


def test_the_legend_texture_renders_and_reports_its_aspect(tmp_path):
    from cognisom.visualization.legend_texture import render_legend_png

    out = tmp_path / "legend.png"
    path, aspect = render_legend_png(
        iusd.legend_entries_for([KIND_KILL, KIND_DIVISION, KIND_ACTIVATION]),
        str(out), subtitle="test", footer="x " * 200)
    assert out.exists() and out.stat().st_size > 0
    assert aspect > 1.0


def test_the_legend_quad_sits_inside_the_camera_frustum():
    stage = _stage()
    info = iusd.author_event_camera(stage, (0.0, 0.0, 0.0), 10.0)
    path = iusd.author_legend(stage, info["path"], "legend.png",
                              focal_length=info["focal_length"],
                              aperture=info["aperture"], texture_aspect=2.0)
    quad = UsdGeom.Mesh.Get(stage, path)
    assert quad.GetPrim().GetParent().GetPath() == \
        UsdGeom.Camera.Get(stage, info["path"]).GetPath()
    pts = [tuple(float(c) for c in p) for p in quad.GetPointsAttr().Get()]
    depth = abs(pts[0][2])
    half_w = depth * (info["aperture"] / 2.0) / info["focal_length"]
    half_h = half_w / (16.0 / 9.0)
    for x, y, _z in pts:
        assert -half_w <= x <= half_w
        assert -half_h <= y <= half_h


def test_the_dome_light_is_not_authored_at_an_intensity_that_blows_out():
    """cells_to_usd authors 500, at which a 0.05 grey renders pure white."""
    stage = _stage()
    dome = iusd.ensure_dome_light(stage)
    assert float(dome.GetIntensityAttr().Get()) <= 2.0


def test_the_stage_timeline_matches_the_frame_range():
    stage = _stage()
    fm = FrameMap([i * 0.2 for i in range(13)])
    iusd.ensure_stage_timeline(stage, fm, fps=24.0)
    assert stage.GetStartTimeCode() == 0.0
    assert stage.GetEndTimeCode() == 12.0
    assert stage.GetTimeCodesPerSecond() == 24.0


def test_the_stage_opens_in_openusd(tmp_path):
    """Everything this layer authors has to survive a save and reload."""
    path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    fm = FrameMap([i * 0.2 for i in range(20)])
    iusd.ensure_stage_timeline(stage, fm)
    iusd.ensure_dome_light(stage)
    iusd.author_interaction_markers(
        stage, [_kill(0.2 * 5, partner=(20.0, 10.0, 10.0))], fm)
    iusd.author_engagement_bonds(
        stage, [_kill(0.2 * 5, partner=(20.0, 10.0, 10.0))], fm)
    positions = np.array([[10.0, 10.0, 10.0]])
    iusd.author_cell_oxygen(
        stage, [(positions, np.array([0.01]))] * 20, fm)
    iusd.author_event_camera(stage, (10.0, 10.0, 10.0), 30.0)
    stage.Save()

    reopened = Usd.Stage.Open(str(path))
    assert reopened is not None
    assert reopened.GetPrimAtPath("/World/Interactions/Kill")
    assert reopened.GetPrimAtPath("/World/Fields/HypoxicCells")
