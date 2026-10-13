"""Turn engine events into a replayable, time-stamped visual record.

Why this exists
---------------
The simulation already emits everything a viewer would want to see -- a
T cell engaging a target, an NK cell killing it, a division, a death, a
hypoxic region appearing -- and none of it was ever drawn. The event bus
delivers each one exactly once, inside the step in which it happened,
and then the information is gone: ``EventBus.event_log`` is a bounded
recency window, ``EventBus.get_trace()`` keeps the closed-loop chain but
carries no simulation time, and neither records *when* in simulated
hours anything occurred.

This recorder is the missing piece. It subscribes to the bus, and for
every event it keeps:

* the simulation time and step, read from the engine at the moment of
  delivery;
* the position **carried in the event payload**, which every spatial
  emitter fills in with ``cell.position.tolist()`` at the instant of
  emission.

That second point is the whole design. A marker drawn at a cell's
*current* position for something that happened forty frames ago is a
lie, and it is an easy lie to tell, because the cell objects are still
sitting there in ``CellularModule.cells`` waiting to be queried. The
recorder therefore never looks a position up after the fact. It copies
the payload's own position, which is a plain list the emitter built from
the cell's coordinates in the step the event fired, and which no later
motion can change.

There is exactly one case where a position is resolved by lookup rather
than read from the payload: the *killer's* position for a kill. The
``CANCER_KILLED`` payload names ``killer_id`` but carries only the
target's coordinates, so the attacker's end of the engagement cannot be
read from the event. A caller may supply ``position_lookup``, which the
recorder calls **synchronously inside the event callback** -- i.e. in
the same engine step the kill was emitted in, before anything has
moved. Those records are tagged
``partner_provenance='live_at_emission'`` so a consumer can tell them
apart from payload-derived coordinates. Without a lookup the attacker's
position is simply absent and nothing is drawn for it; see
``interaction_usd.author_engagement_bonds``.

Events that carry no position at all are kept too, and kept distinct.
``HYPOXIA_DETECTED`` reports a count and a severity and no coordinates,
so it is recorded with ``position=None`` and surfaces through
:attr:`InteractionRecorder.global_events`. Nothing invents a location
for it. Hypoxia is made visible instead by drawing the oxygen field,
which is a solved quantity, rather than by guessing where the event
happened.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

try:  # numpy is a hard dependency of the engine, but the recorder
    import numpy as _np  # itself only needs it to normalise arrays.
except Exception:  # pragma: no cover - numpy is always present in-tree
    _np = None


Vec3 = Tuple[float, float, float]


# ── Event kinds ──────────────────────────────────────────────────────
#
# A "kind" is the visual vocabulary: one glyph, one colour, one meaning.
# It is deliberately coarser than EventTypes. ``cell_killed_by_tcell``
# and ``cancer_killed`` are the same picture -- a kill -- and are drawn
# once, not twice; the record keeps the raw event type so a consumer can
# still tell an adaptive kill from an innate one.

KIND_KILL = "kill"
KIND_ACTIVATION = "activation"
KIND_DIVISION = "division"
KIND_DEATH = "death"
KIND_TRANSFORMATION = "transformation"
KIND_RECRUITMENT = "recruitment"
KIND_METASTASIS = "metastasis"
KIND_HYPOXIA = "hypoxia"
KIND_EXHAUSTION = "exhaustion"

#: Event type -> visual kind. Anything not listed here is ignored by the
#: recorder: ``gene_expressed`` fires 1662 times in a six-hour run and
#: ``circadian_peak`` 47 times, neither has a position, and drawing them
#: would bury the four events a viewer actually came to see.
DEFAULT_EVENT_KINDS: Dict[str, str] = {
    "cancer_killed": KIND_KILL,
    "cell_killed_by_tcell": KIND_KILL,
    "immune_activated": KIND_ACTIVATION,
    "cell_divided": KIND_DIVISION,
    "cell_died": KIND_DEATH,
    "cell_transformed": KIND_TRANSFORMATION,
    "immune_recruited": KIND_RECRUITMENT,
    "metastasis_occurred": KIND_METASTASIS,
    "hypoxia_detected": KIND_HYPOXIA,
    "tcell_exhausted": KIND_EXHAUSTION,
}

#: Deaths whose cause is an immune kill are already drawn, at the same
#: coordinates and the same instant, by the kill marker. Recording both
#: is correct -- they are two different facts -- but drawing both puts
#: two glyphs on one pixel. The writer filters on this set; the recorder
#: keeps everything.
IMMUNE_KILL_DEATH_CAUSES = frozenset({"immune_killed"})


def _as_vec3(value: Any) -> Optional[Vec3]:
    """Coerce an event payload's position to a plain 3-tuple of floats.

    Emitters hand over ``list`` (from ``ndarray.tolist()``), and nothing
    else is accepted silently: a 2-vector or a scalar is a malformed
    payload and returning a padded guess would put a marker at a
    coordinate the simulation never produced.
    """
    if value is None:
        return None
    if _np is not None and isinstance(value, _np.ndarray):
        value = value.tolist()
    try:
        seq = list(value)
    except TypeError:
        return None
    if len(seq) != 3:
        return None
    try:
        return (float(seq[0]), float(seq[1]), float(seq[2]))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class VisualEvent:
    """One thing that happened, where and when it happened.

    ``position`` is the coordinate the emitter put in the payload, which
    is the cell's position in the step the event fired. It is never
    refreshed. ``provenance`` says so explicitly, so a consumer reading
    a serialised record can tell a recorded coordinate from one that was
    looked up.
    """

    kind: str
    event_type: str
    time: float
    step: int
    position: Optional[Vec3] = None
    cell_id: Optional[int] = None
    partner_id: Optional[int] = None
    partner_position: Optional[Vec3] = None
    label: str = ""
    #: 'event_payload' when the position came from the emitted data,
    #: 'none' when the event carried no coordinates at all.
    provenance: str = "event_payload"
    #: 'live_at_emission' when partner_position was resolved by calling
    #: position_lookup inside the event callback (same step, nothing has
    #: moved), 'none' when no partner position is known.
    partner_provenance: str = "none"
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_position(self) -> bool:
        return self.position is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "event_type": self.event_type,
            "time": self.time,
            "step": self.step,
            "position": list(self.position) if self.position else None,
            "cell_id": self.cell_id,
            "partner_id": self.partner_id,
            "partner_position": (
                list(self.partner_position) if self.partner_position else None
            ),
            "label": self.label,
            "provenance": self.provenance,
            "partner_provenance": self.partner_provenance,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisualEvent":
        return cls(
            kind=d["kind"],
            event_type=d["event_type"],
            time=float(d["time"]),
            step=int(d["step"]),
            position=_as_vec3(d.get("position")),
            cell_id=d.get("cell_id"),
            partner_id=d.get("partner_id"),
            partner_position=_as_vec3(d.get("partner_position")),
            label=d.get("label", ""),
            provenance=d.get("provenance", "event_payload"),
            partner_provenance=d.get("partner_provenance", "none"),
            detail=d.get("detail", {}) or {},
        )


@dataclass
class Engagement:
    """An activation and the kill it led to, matched by (killer, target).

    Both coordinates are real recordings, but they are recordings from
    *different* instants, which is exactly why this is reported as an
    interval rather than drawn as a static line. In a measured 48 h
    Immunotherapy run one engagement ran from t=11.95 h to t=30.00 h --
    eighteen hours. A bond drawn between those two points for eighteen
    hours would be a fabrication for all but its last frame.
    """

    killer_id: Optional[int]
    target_id: Optional[int]
    activated_at: float
    killed_at: float
    activation_position: Optional[Vec3]
    kill_position: Optional[Vec3]

    @property
    def duration_hours(self) -> float:
        return self.killed_at - self.activated_at


class InteractionRecorder:
    """Subscribe to an EventBus and keep a time-stamped visual record.

    Parameters
    ----------
    time_fn:
        Returns the current simulation time in hours. Called once per
        delivered event, inside the callback.
    step_fn:
        Returns the current step index. Optional.
    position_lookup:
        ``f(cell_id) -> position | None``, used for the *killer's*
        coordinates on a kill and for nothing else. It is called inside
        the event callback, so it reads the attacker's position in the
        same step the kill was emitted. Pass ``None`` (the default) and
        no attacker coordinate is recorded, which means no engagement
        bond is drawn -- an absent line rather than a wrong one.
    event_kinds:
        Override the event-type -> kind mapping.
    max_events:
        Hard cap. Beyond it the recorder stops appending and counts the
        overflow in :attr:`dropped`, rather than growing without bound
        in a long run. Nothing is evicted: the head of a record is the
        part that cannot be reconstructed, the same reasoning as
        ``EventBus.retained_log``.
    """

    def __init__(
        self,
        time_fn: Callable[[], float],
        step_fn: Optional[Callable[[], int]] = None,
        position_lookup: Optional[Callable[[Any], Any]] = None,
        event_kinds: Optional[Dict[str, str]] = None,
        max_events: int = 200_000,
    ):
        self._time_fn = time_fn
        self._step_fn = step_fn or (lambda: -1)
        self._position_lookup = position_lookup
        self.event_kinds = dict(
            DEFAULT_EVENT_KINDS if event_kinds is None else event_kinds
        )
        self.max_events = int(max_events)

        self.events: List[VisualEvent] = []
        self.dropped = 0
        #: Field snapshots, ``name -> [(time, ndarray), ...]``. Filled by
        #: :meth:`capture_field`; the recorder never goes and fetches
        #: them, because when they are captured is the caller's business.
        self.field_frames: Dict[str, List[Tuple[float, Any]]] = {}
        self.field_meta: Dict[str, Dict[str, Any]] = {}

        self._bus = None
        self._handlers: List[Tuple[str, Callable]] = []

    # ── Wiring ───────────────────────────────────────────────────────

    def attach(self, bus) -> "InteractionRecorder":
        """Subscribe to every event type in the kind map."""
        if self._bus is not None:
            raise RuntimeError(
                "InteractionRecorder is already attached to an event bus. "
                "Attaching twice would record every event twice."
            )
        self._bus = bus
        for event_type in self.event_kinds:
            handler = self._make_handler(event_type)
            bus.subscribe(event_type, handler)
            self._handlers.append((event_type, handler))
        return self

    def detach(self) -> None:
        """Unsubscribe. Safe to call when never attached."""
        if self._bus is None:
            return
        for event_type, handler in self._handlers:
            self._bus.unsubscribe(event_type, handler)
        self._handlers.clear()
        self._bus = None

    @classmethod
    def for_engine(cls, engine, **kwargs) -> "InteractionRecorder":
        """Build a recorder wired to a running :class:`SimulationEngine`.

        The default ``position_lookup`` resolves an immune cell id to its
        current position through the engine's immune module. It is only
        ever consulted during event delivery, so "current" means "in the
        step the event fired".
        """
        immune = None
        try:
            immune = engine.modules.get("immune")
        except Exception:
            immune = None

        lookup = kwargs.pop("position_lookup", "default")
        if lookup == "default":
            if immune is None:
                lookup = None
            else:
                def lookup(cell_id, _immune=immune):  # noqa: E306
                    cell = getattr(_immune, "immune_cells", {}).get(cell_id)
                    return None if cell is None else getattr(cell, "position", None)

        rec = cls(
            time_fn=lambda: float(getattr(engine, "time", 0.0)),
            step_fn=lambda: int(getattr(engine, "step_count", -1)),
            position_lookup=lookup,
            **kwargs,
        )
        rec.attach(engine.event_bus)
        return rec

    # ── Recording ────────────────────────────────────────────────────

    def _make_handler(self, event_type: str) -> Callable:
        def handler(data):
            self._record(event_type, data)
        handler.__qualname__ = f"InteractionRecorder[{event_type}]"
        return handler

    def _record(self, event_type: str, data: Any) -> None:
        if len(self.events) >= self.max_events:
            self.dropped += 1
            return

        payload = data if isinstance(data, dict) else {}
        kind = self.event_kinds.get(event_type, event_type)

        position = _as_vec3(payload.get("position"))
        provenance = "event_payload" if position is not None else "none"

        cell_id = payload.get("cell_id", payload.get("immune_id"))
        partner_id = payload.get(
            "killer_id", payload.get("target_id", payload.get("daughter_id"))
        )

        partner_position = None
        partner_provenance = "none"
        if kind == KIND_KILL and partner_id is not None and self._position_lookup:
            # Resolved HERE, synchronously, inside the callback: the
            # engine has not stepped since the kill was emitted, so this
            # is the attacker's position at the instant of the kill and
            # not a stale read from some later frame.
            try:
                partner_position = _as_vec3(self._position_lookup(partner_id))
            except Exception:
                log.debug("position_lookup failed for id %r", partner_id)
                partner_position = None
            if partner_position is not None:
                partner_provenance = "live_at_emission"

        self.events.append(
            VisualEvent(
                kind=kind,
                event_type=event_type,
                time=float(self._time_fn()),
                step=int(self._step_fn()),
                position=position,
                cell_id=cell_id,
                partner_id=partner_id,
                partner_position=partner_position,
                label=_label_for(kind, event_type, payload),
                provenance=provenance,
                partner_provenance=partner_provenance,
                detail=_detail_for(event_type, payload),
            )
        )

    def capture_field(
        self,
        name: str,
        array: Any,
        time: float,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        resolution: float = 1.0,
    ) -> None:
        """Store a copy of a scalar field at one instant.

        A copy, not a reference: the engine mutates its field arrays in
        place, so keeping the reference would give every frame the final
        state of the run -- an animation in which the oxygen field never
        changes.
        """
        if _np is None:  # pragma: no cover
            raise RuntimeError("capture_field requires numpy")
        arr = _np.array(array, dtype=_np.float32, copy=True)
        self.field_frames.setdefault(name, []).append((float(time), arr))
        self.field_meta[name] = {
            "origin": tuple(float(v) for v in origin),
            "resolution": float(resolution),
            "shape": tuple(int(n) for n in arr.shape),
        }

    def capture_engine_fields(
        self, engine, time: Optional[float] = None,
        names: Sequence[str] = ("oxygen",),
    ) -> None:
        """Convenience: snapshot named SpatialModule fields.

        Does nothing when the spatial module is absent or a named field
        does not exist -- a scenario with no spatial module should not
        raise here, it should simply render without a field.
        """
        spatial = None
        try:
            spatial = engine.modules.get("spatial")
        except Exception:
            spatial = None
        if spatial is None:
            return
        t = float(getattr(engine, "time", 0.0)) if time is None else float(time)
        for name in names:
            fld = getattr(spatial, "fields", {}).get(name)
            if fld is None:
                continue
            self.capture_field(
                name,
                fld.concentration,
                t,
                origin=(0.0, 0.0, 0.0),
                resolution=float(getattr(spatial, "resolution", 1.0)),
            )

    # ── Reading back ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.events)

    @property
    def positioned_events(self) -> List[VisualEvent]:
        """Events that can be drawn at a place."""
        return [e for e in self.events if e.has_position]

    @property
    def global_events(self) -> List[VisualEvent]:
        """Events with no coordinates -- ``hypoxia_detected`` and friends.

        These are kept and reported, never placed. Giving them a
        position would mean inventing one.
        """
        return [e for e in self.events if not e.has_position]

    def of_kind(self, *kinds: str) -> List[VisualEvent]:
        wanted = set(kinds)
        return [e for e in self.events if e.kind in wanted]

    def between(self, t0: float, t1: float) -> List[VisualEvent]:
        """Events with ``t0 <= time <= t1``."""
        return [e for e in self.events if t0 <= e.time <= t1]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.events:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out

    def time_span(self) -> Tuple[float, float]:
        if not self.events:
            return (0.0, 0.0)
        times = [e.time for e in self.events]
        return (min(times), max(times))

    def engagements(self) -> List[Engagement]:
        """Match each kill back to the activation that preceded it.

        Pairs on ``(killer id, target id)`` and takes the most recent
        unconsumed activation for that pair. Used for reporting how long
        engagements last -- which is the number that decides whether an
        engagement bond can honestly be drawn -- not for placing
        geometry.
        """
        pending: Dict[Tuple[Any, Any], List[VisualEvent]] = {}
        out: List[Engagement] = []
        for e in self.events:
            if e.kind == KIND_ACTIVATION:
                key = (e.cell_id, e.detail.get("target_id"))
                pending.setdefault(key, []).append(e)
            elif e.kind == KIND_KILL:
                key = (e.partner_id, e.cell_id)
                queue = pending.get(key)
                act = queue.pop() if queue else None
                out.append(
                    Engagement(
                        killer_id=e.partner_id,
                        target_id=e.cell_id,
                        activated_at=act.time if act else e.time,
                        killed_at=e.time,
                        activation_position=act.position if act else None,
                        kill_position=e.position,
                    )
                )
        return out

    # ── Serialisation ────────────────────────────────────────────────

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    def write_json(self, path: str) -> str:
        payload = {
            "version": 1,
            "generator": "cognisom.visualization.interaction_recorder",
            "counts": self.counts(),
            "dropped": self.dropped,
            "events": self.to_dicts(),
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=1)
        return path

    @staticmethod
    def events_from_dicts(dicts: Iterable[Dict[str, Any]]) -> List[VisualEvent]:
        return [VisualEvent.from_dict(d) for d in dicts]


# ── Labelling ────────────────────────────────────────────────────────

def _label_for(kind: str, event_type: str, payload: Dict[str, Any]) -> str:
    """A short human string, for the manifest and for debugging."""
    if kind == KIND_KILL:
        killer = payload.get("killer_type") or (
            "T_cell" if event_type == "cell_killed_by_tcell" else "immune"
        )
        return f"{killer} killed cell {payload.get('cell_id')}"
    if kind == KIND_ACTIVATION:
        return (
            f"{payload.get('immune_type', 'immune')} "
            f"{payload.get('immune_id')} engaged {payload.get('target_id')}"
        )
    if kind == KIND_DIVISION:
        return (
            f"{payload.get('cell_type', 'cell')} {payload.get('cell_id')} "
            f"-> {payload.get('daughter_id')}"
        )
    if kind == KIND_DEATH:
        return (
            f"{payload.get('cell_type', 'cell')} {payload.get('cell_id')} "
            f"died ({payload.get('cause', 'unknown')})"
        )
    if kind == KIND_HYPOXIA:
        return (
            f"{payload.get('n_hypoxic_cells', '?')} hypoxic cells "
            f"({payload.get('severity', '?')})"
        )
    if kind == KIND_TRANSFORMATION:
        return f"cell {payload.get('cell_id')} transformed"
    if kind == KIND_RECRUITMENT:
        return f"{payload.get('cell_type', 'immune')} recruited"
    return event_type


_DETAIL_KEYS = (
    "cell_type", "killer_type", "immune_type", "cause", "target_id",
    "immune_id", "daughter_id", "severity", "n_hypoxic_cells", "tcr_id",
    "affinity", "peptide", "hla_allele", "mutation", "source_gene",
    "reason", "encounter_count",
)


def _detail_for(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the scalar fields a viewer or a caption might want.

    Deliberately a whitelist: payloads also carry whole mutation lists
    and nested objects, and a visual record that drags those along stops
    being serialisable.
    """
    out = {}
    for key in _DETAIL_KEYS:
        if key in payload:
            value = payload[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
    return out
