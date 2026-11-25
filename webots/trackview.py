"""Show on the 3D view how the chair predicts people will walk, and what kind of walker each is
(T in webots/run.py, or --tracks).

For each person the chair's tracker follows (the nearest few):
  * an orange ribbon: the track the chair predicts for the next 3 s (assist/tracker.py: they keep
    walking as they have been over the last second),
  * a blue ribbon: the way they have walked in the last 3 s,
  * a thin black line on top of the orange ribbon: where they really walk next 3 s, for people replaying a recorded track
    (replay.py), so the prediction can be compared with what happened,
  * a ring on the floor, coloured by what they are like (envs/behaviours.py): green walking,
    magenta on the phone, blue in a group, yellow stopped or waiting, purple older, white standing,
    orange something else (a child, a pet, a trolley).

It also scores the prediction as SiT does for trajectory prediction: every half second each
tracked person's predicted positions 0.5 to 3 s ahead are kept, and when that time comes they are
compared with where the person really is. ADE is the mean error over all of them (m), FDE the mean
error at 3 s. `summary()` has the running numbers; `reset()` starts them again for a new trip.

Like the sensor view these are runtime-only nodes, shown on the 3D view only, with no physics and
no bounding object, so nothing else in the simulator changes.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

SLOTS = 10                         # people drawn at once (the nearest ones)
HORIZONS_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
PAST_S, PAST_STEP_S = 3.0, 0.5
SCORE_EVERY = 5                    # steps between the predictions kept for scoring
HALF_WIDTH_M = 0.06
MIN_SEGMENT_M = 0.03
MOVING_MPS = 0.15                  # slower than this is standing still: no ribbon
HIDDEN_Z = -1000.0
BEHAVIOURS = {                     # name: ring colour (red, green, blue)
    "walking": (0.10, 0.85, 0.20),
    "phone": (0.95, 0.15, 0.85),
    "group": (0.15, 0.35, 0.95),
    "stopped": (0.95, 0.85, 0.10),
    "elderly": (0.60, 0.25, 0.85),
    "standing": (0.90, 0.90, 0.90),
    "other": (1.00, 0.50, 0.00),
    "replay": (0.02, 0.02, 0.02),   # walking a recorded (SiT) track
}
PRED, PAST, TRUTH = (0.95, 0.50, 0.10), (0.20, 0.40, 1.00), (0.05, 0.05, 0.05)


def behaviour_of(person) -> str:
    """What kind of walker `person` is right now (a key of BEHAVIOURS)."""
    from envs import behaviours
    if person.kind != "person":
        return "other"
    if getattr(person, "_replay", None) is not None:
        return "replay"
    t = behaviours.traits_of(person)
    if getattr(person, "type", "dynamic") == "static" and person.speed < 0.05:
        return "standing"
    if t is not None and (t.pause_left > 0 or t.dwell_left is not None):
        return "stopped"
    if person.speed < 0.05 and person.player is None and getattr(person, "type", "dynamic") == "dynamic":
        return "stopped"
    if t is not None and t.phone:
        return "phone"
    if t is not None and (t.leader is not None or t.members):
        return "group"
    if person.elderly:
        return "elderly"
    return "walking"


def ribbon(samples, half_width: float = HALF_WIDTH_M, z: float = 0.02):
    """The corners of a flat strip along `samples` ((x, y) points, at least 2): for each sample the
    point to its left, then to its right, as a list of [x, y, z]. Samples closer than MIN_SEGMENT_M to
    the one before are moved on, so that no triangle of the strip is flat. Returns None if it cannot
    be drawn (a single point, or no distance covered)."""
    pts = [np.asarray(samples[0], float)]
    heading = None
    for q in samples[1:]:
        q = np.asarray(q, float)
        d = q - pts[-1]
        n = float(np.hypot(*d))
        if n >= 1e-9:
            heading = d / n
        if n < MIN_SEGMENT_M:
            if heading is None:
                continue
            q = pts[-1] + heading * MIN_SEGMENT_M
        pts.append(q)
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    d = np.gradient(pts, axis=0)
    norm = np.hypot(d[:, 0], d[:, 1])[:, None]
    d = d / np.where(norm < 1e-9, 1.0, norm)
    left = np.stack([-d[:, 1], d[:, 0]], axis=1) * half_width
    corners = []
    for p, l in zip(pts, left):
        corners += [[float(p[0] + l[0]), float(p[1] + l[1]), z], [float(p[0] - l[0]), float(p[1] - l[1]), z]]
    return corners


def faces(n_samples: int) -> str:
    """coordIndex of a strip of n_samples left/right pairs: counter-clockwise from above, so it faces up."""
    return " ".join(f"{2 * i} {2 * i + 1} {2 * i + 3} {2 * i + 2} -1" for i in range(n_samples - 1))


def hidden_points(n_samples: int) -> list[list[float]]:
    """A strip far below the floor that is not flat (a flat triangle makes Webots warn)."""
    return [[0.1 * (i // 2), 0.05 * (1 if i % 2 == 0 else -1), HIDDEN_Z] for i in range(2 * n_samples)]


def strip_node(name: str, n_samples: int, colour, transparency: float = 0.15) -> str:
    r, g, b = colour
    points = " ".join(f"{v:g}" for p in hidden_points(n_samples) for v in p)
    return f"""DEF {name}_POSE Pose {{ children [ Shape {{
  appearance PBRAppearance {{ baseColor {r} {g} {b} emissiveColor {r} {g} {b} emissiveIntensity 0.7 transparency {transparency} roughness 1 metalness 0 }}
  geometry IndexedFaceSet {{
    coord DEF {name}_POINTS Coordinate {{ point [ {points} ] }}
    coordIndex [ {faces(n_samples)} ]
    creaseAngle 0
  }}
}} ] }}"""


def ring_node(i: int) -> str:
    r, g, b = BEHAVIOURS["walking"]
    return f"""DEF TRACK_RING{i} Pose {{
  translation 0 0 {HIDDEN_Z:g}
  children [ Shape {{
    appearance DEF TRACK_RING{i}_LOOK PBRAppearance {{ baseColor {r} {g} {b} emissiveColor {r} {g} {b} emissiveIntensity 0.6 transparency 0.45 roughness 1 metalness 0 }}
    geometry Cylinder {{ height 0.01 radius 0.38 }} }} ]
}}"""


class TrackView:
    """The ribbons, the rings and the prediction score. `update(tracks)` every step."""

    def __init__(self, scene):
        self.scene = scene
        sup = scene.sup
        self.n_pred, self.n_past, self.n_truth = len(HORIZONS_S) + 1, int(PAST_S / PAST_STEP_S) + 1, len(HORIZONS_S) + 1
        if sup.getFromDef("TRACK_RING0") is None:
            children = sup.getRoot().getField("children")
            for i in range(SLOTS):
                for name, n, colour in ((f"TRACK_PRED{i}", self.n_pred, PRED), (f"TRACK_PAST{i}", self.n_past, PAST),
                                        (f"TRACK_TRUTH{i}", self.n_truth, TRUTH)):
                    children.importMFNodeFromString(-1, strip_node(name, n, colour))
                children.importMFNodeFromString(-1, ring_node(i))
        self.pred, self.past, self.truth, self.rings, self.looks = [], [], [], [], []
        for i in range(SLOTS):
            self.pred.append(sup.getFromDef(f"TRACK_PRED{i}_POINTS").getField("point"))
            self.past.append(sup.getFromDef(f"TRACK_PAST{i}_POINTS").getField("point"))
            self.truth.append(sup.getFromDef(f"TRACK_TRUTH{i}_POINTS").getField("point"))
            self.rings.append(sup.getFromDef(f"TRACK_RING{i}").getField("translation"))
            self.looks.append(sup.getFromDef(f"TRACK_RING{i}_LOOK"))
        for i in range(SLOTS):
            for name in (f"TRACK_PRED{i}_POSE", f"TRACK_PAST{i}_POSE", f"TRACK_TRUTH{i}_POSE", f"TRACK_RING{i}"):
                scene.only_on_screen(sup.getFromDef(name))
        self.on = True
        self.drawn = [False] * SLOTS          # which strips show a real track now (the rest are hidden)
        self._colour = [None] * SLOTS
        self.reset()

    def reset(self) -> None:
        """A new trip: forget the trails and the scores."""
        self.trails: dict[int, deque] = {}
        self.pending: list[dict] = []
        self.errors = {h: [] for h in HORIZONS_S}
        self.counts: dict[str, int] = {}

    # ------------------------------------------------------------ drawing
    def show(self, on: bool) -> None:
        self.on = on
        if not on:
            for i in range(SLOTS):
                self._hide(i)

    def _hide(self, i: int) -> None:
        if not any((self.drawn[i], self._colour[i])):
            return
        for field, n in ((self.pred[i], self.n_pred), (self.past[i], self.n_past), (self.truth[i], self.n_truth)):
            for k, point in enumerate(hidden_points(n)):
                field.setMFVec3f(k, point)
        self.rings[i].setSFVec3f([0.0, 0.0, HIDDEN_Z])
        self.drawn[i], self._colour[i] = False, None

    @staticmethod
    def _set(field, points) -> None:
        for k, point in enumerate(points):
            field.setMFVec3f(k, point)

    def _strip(self, field, samples, n: int, half_width: float = HALF_WIDTH_M, z: float = 0.02) -> None:
        """Draw `samples` on `field` if it can be drawn and has the right number of points, else hide it."""
        corners = ribbon(samples, half_width, z) if len(samples) >= 2 else None
        if corners is None or len(corners) != 2 * n:
            corners = hidden_points(n)
        self._set(field, corners)

    def update(self, tracks) -> None:
        """Redraw from the chair's tracker's `tracks` (assist.tracker.Track, nearest first) and score them."""
        scene = self.scene
        dt = scene.TIMESTEP
        now = scene.ticks * dt
        people = {p.id: p for p in scene.crowd.everyone}
        for track in tracks:                                    # the last few seconds of everyone seen
            person = people.get(track.id)
            if person is not None:
                self.trails.setdefault(track.id, deque(maxlen=int(PAST_S / dt) + 1)).append((person.x, person.y))
        self._score(now, tracks, people)
        if not self.on:
            return
        shown = 0
        for track in tracks:
            person = people.get(track.id)
            if person is None or shown >= SLOTS:
                continue
            i = shown
            shown += 1
            speed = math.hypot(track.vx, track.vy)
            if speed >= MOVING_MPS:
                line = [(track.x + track.vx * h, track.y + track.vy * h) for h in (0.0,) + HORIZONS_S]
                self._strip(self.pred[i], line, self.n_pred)
            else:
                self._strip(self.pred[i], [], self.n_pred)
            trail = list(self.trails.get(track.id, ()))
            step = max(1, int(round(PAST_STEP_S / dt)))
            samples = trail[::-1][::step][::-1] if len(trail) > step else []
            self._strip(self.past[i], samples[-self.n_past:], self.n_past)
            replay = getattr(person, "_replay", None)
            real = replay.future(now, HORIZONS_S[-1], HORIZONS_S[1] - HORIZONS_S[0]) if replay is not None else []
            if real:                                        # as far as the recording goes; the end is repeated (and nudged on)
                real = real + [real[-1]] * (self.n_truth - 1 - len(real))
            self._strip(self.truth[i], [(person.x, person.y)] + real if real else [], self.n_truth,
                        half_width=0.7 * HALF_WIDTH_M, z=0.02)      # a thin black line on top of the orange ribbon
            kind = behaviour_of(person)
            if kind != self._colour[i]:
                r, g, b = BEHAVIOURS[kind]
                self.looks[i].getField("baseColor").setSFColor([r, g, b])
                self.looks[i].getField("emissiveColor").setSFColor([r, g, b])
                self._colour[i] = kind
            self.rings[i].setSFVec3f([person.x, person.y, 0.012 + 0.0005 * i])
            self.drawn[i] = True
        for i in range(shown, SLOTS):
            self._hide(i)

    # ------------------------------------------------------------ scoring
    def _score(self, now: float, tracks, people) -> None:
        """Keep each tracked person's predicted positions every SCORE_EVERY steps, and compare the ones
        that have come due with where the person really is."""
        dt = self.scene.TIMESTEP
        if self.scene.ticks % SCORE_EVERY == 0:
            for t in tracks:
                p = people.get(t.id)
                if p is None:
                    continue
                self.pending.append({"id": t.id, "t0": now, "done": 0,
                                     "pred": [(t.x + t.vx * h, t.y + t.vy * h) for h in HORIZONS_S]})
                self.counts[behaviour_of(p)] = self.counts.get(behaviour_of(p), 0) + 1
        keep = []
        for entry in self.pending:
            p = people.get(entry["id"])
            while entry["done"] < len(HORIZONS_S) and now + 1e-6 >= entry["t0"] + HORIZONS_S[entry["done"]]:
                if p is not None:
                    px, py = entry["pred"][entry["done"]]
                    self.errors[HORIZONS_S[entry["done"]]].append(math.hypot(p.x - px, p.y - py))
                entry["done"] += 1
            if entry["done"] < len(HORIZONS_S) and p is not None:
                keep.append(entry)
        self.pending = keep

    def scores(self) -> tuple[float, float, int]:
        """(ADE, FDE, number of scored predictions) so far this trip; nan if none yet."""
        every = [e for h in HORIZONS_S for e in self.errors[h]]
        last = self.errors[HORIZONS_S[-1]]
        if not last:
            return float("nan"), float("nan"), 0
        return float(np.mean(every)), float(np.mean(last)), len(last)

    def summary(self) -> str:
        ade, fde, n = self.scores()
        if not n:
            return "Prediction: scoring..."
        return f"Prediction error ADE {ade:.2f} m, FDE (3 s) {fde:.2f} m over {n} predictions"
