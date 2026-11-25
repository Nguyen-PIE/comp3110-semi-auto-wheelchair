"""Show the proxemic zones around the chair on the 3D view (Z in webots/run.py, or --zones).

The intimate, personal and social zones of assist/proxemics.py, drawn on the floor as see-through
rings with no outline: red up to 0.45 m from the chair's outline (intimate), orange to 1.2 m
(personal) and yellow to 3.6 m (social). Everything further out is the public zone. The front of
every ring stretches as the chair speeds up (up to twice as deep at full speed), as the proxemic
reward and the zone measures in eval/comfort.py see them. A person is in the zone their nearest edge
reaches, as in assist/proxemics.py.

The rings move with the chair on every physics frame (bridge.py's `every_frame`, as the rider is
seated), so they do not lag behind it between the 0.1 s steps. The back half is drawn once and the
front half for 41 speeds, and the front shown follows the chair's speed smoothed over about 0.3 s,
so it grows and shrinks gradually. Like the sensor view these are runtime-only nodes, shown on the 3D
view only, with no physics and no bounding object, and hidden from the chair's camera and the phone
walkers' eyes, so nothing else in the simulator changes.

They are never shown with the sensor fan (sensorview.py), which is see-through on the same floor:
where two see-through layers overlap, Webots draws them in an order that changes as they move, and
the colours flicker. run.py shows one at a time.
"""

from __future__ import annotations

import math

from assist import proxemics as px

LEVELS = 41                        # front halves drawn, for speeds 0, 1/40, ... full speed
POINTS = 41                        # points along each half of an ellipse (odd, so the very front is one)
SMOOTH_S = 0.3                     # the speed the rings show follows the chair's over about this long
Z_M = 0.03                         # just above the floor
HIDDEN_Z = -1000.0
# zone, from (m), to (m), colour, how see-through (0 solid, 1 invisible)
RINGS = (
    ("intimate", 0.0, px.ZONE_EDGES_M[0], (0.92, 0.33, 0.25), 0.25),
    ("personal", px.ZONE_EDGES_M[0], px.ZONE_EDGES_M[1], (0.96, 0.62, 0.16), 0.40),
    ("social", px.ZONE_EDGES_M[1], px.ZONE_EDGES_M[2], (0.95, 0.90, 0.30), 0.65),
)


def half(d: float, stretch: float, front: bool) -> list[tuple[float, float]]:
    """Half the edge of a zone `d` m out from the chair's outline, in the chair's frame (x forward,
    y left), anticlockwise seen from above: the front half `stretch` times as deep, or the back half."""
    start = -math.pi / 2 if front else math.pi / 2
    deep = d * stretch if front else d
    pts = []
    for i in range(POINTS):
        t = start + math.pi * i / (POINTS - 1)
        pts.append(((px.HALF_LENGTH_M + deep) * math.cos(t), (px.HALF_WIDTH_M + d) * math.sin(t)))
    return pts


def outline(d: float, stretch: float) -> list[tuple[float, float]]:
    """The whole edge of a zone (front half, then back half)."""
    return half(d, stretch, True) + half(d, stretch, False)


def _band(stretch: float, front: bool) -> str:
    """One half of every ring as see-through strips facing up, each zone its own colour."""
    shapes = []
    for _name, d_in, d_out, (r, g, b), see_through in RINGS:
        outer, inner = half(d_out, stretch, front), half(d_in, stretch, front)
        n = len(outer)
        coords = ", ".join(f"{x:.3f} {y:.3f} {Z_M}" for x, y in outer + inner)
        quads = " ".join(f"{i} {i + 1} {n + i + 1} {n + i} -1" for i in range(n - 1))
        shapes.append(
            f"Shape {{ castShadows FALSE appearance Appearance {{ material Material {{ diffuseColor {r} {g} {b} "
            f"emissiveColor {r * 0.3:.3f} {g * 0.3:.3f} {b * 0.3:.3f} transparency {see_through} }} }} "
            f"geometry IndexedFaceSet {{ coord Coordinate {{ point [ {coords} ] }} coordIndex [ {quads} ] }} }}")
    return " ".join(shapes)


def zones_node() -> str:
    fronts = " ".join(f"DEF ZONES_F{k} Pose {{ translation 0 0 {HIDDEN_Z:g} children [ "
                      f"{_band(1 + px.MAX_STRETCH * k / (LEVELS - 1), True)} ] }}" for k in range(LEVELS))
    return (f"DEF ZONES Pose {{ translation 0 0 {HIDDEN_Z:g} children [ "
            f"Pose {{ children [ {_band(1.0, False)} ] }} {fronts} ] }}")


class ZoneView:
    """The zone rings. They follow the chair by themselves; `show()` turns them on and off."""

    def __init__(self, scene):
        self.scene = scene
        sup = scene.sup
        if sup.getFromDef("ZONES") is None:
            sup.getRoot().getField("children").importMFNodeFromString(-1, zones_node())
        node = sup.getFromDef("ZONES")
        self.position = node.getField("translation")
        self.rotation = node.getField("rotation")
        self.fronts = [sup.getFromDef(f"ZONES_F{k}").getField("translation") for k in range(LEVELS)]
        scene.only_on_screen(node)
        self.on = False
        self.level = None
        self.share = 0.0
        self.every_frame = hasattr(scene, "every_frame")
        if self.every_frame:
            scene.every_frame.append(self.follow)

    def show(self, on: bool) -> None:
        self.on = on
        if on:
            self.follow(0)
        else:
            self.position.setSFVec3f([0.0, 0.0, HIDDEN_Z])

    def update(self) -> None:
        """For a scene without `every_frame`: move the rings once a 0.1 s step."""
        if not self.every_frame:
            self.follow(self.scene.step_ms)

    def follow(self, ms: int) -> None:
        """Put the rings where the chair will be after the next `ms` of simulation, as the rider is."""
        if not self.on:
            return
        chair = self.scene.chair_node
        p, o, v = chair.getPosition(), chair.getOrientation(), chair.getVelocity()
        dt = ms / 1000
        heading = math.atan2(o[3], o[0]) + v[5] * dt
        forward = v[0] * math.cos(heading) + v[1] * math.sin(heading)
        share = min(1.0, max(0.0, forward / self.scene.MAX_ADVANCE_ROBOT))
        self.share += (share - self.share) * (1 - math.exp(-dt / SMOOTH_S)) if dt > 0 else 0.0
        k = round(self.share * (LEVELS - 1))
        if k != self.level:
            if self.level is not None:
                self.fronts[self.level].setSFVec3f([0.0, 0.0, HIDDEN_Z])
            self.fronts[k].setSFVec3f([0.0, 0.0, 0.0])
            self.level = k
        self.position.setSFVec3f([p[0] + v[0] * dt, p[1] + v[1] * dt, 0.0])
        self.rotation.setSFRotation([0.0, 0.0, 1.0, heading])
