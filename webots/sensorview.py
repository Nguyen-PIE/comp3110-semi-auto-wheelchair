"""Show on the 3D view what the chair's sensors can see (S in webots/run.py, or --sensors).

A real wheelchair cannot see round corners or through walls, so a translucent fan is drawn on the
floor around the chair, out to wherever each lidar ray stops (a wall, a coffee table, a person).
Only the 3D view shows it: the people walked from phones (phones.py) and the chair's own camera do not
see it, as nobody else sees what a chair's sensors see.
It is the chair's own lidar, read every step, so it also shows the things a rider can look over:
a low table cuts a notch in the fan even though the rider sees the wall behind it. A green ring
marks every person the assistant's tracker can currently see (the ones in line of sight).

Like the landmarks these are runtime-only nodes with no physics and no bounding object, so the
lidar does not see them and they change nothing about how the chair or the people move.
"""

from __future__ import annotations

import math

RAYS_SHOWN = 90                   # the lidar has 360 or so; this many are drawn
FAN_Z = 0.03                      # just above the floor
PEOPLE_RINGS = 16
HIDDEN = [0.0, 0.0, -1000.0]


def hidden_fan(n: int) -> list[list[float]]:
    """The fan's points while it is hidden: a 1 cm fan far below the floor. Its triangles must not
    be flat, or Webots warns that the mesh has no valid triangle."""
    rim = [[0.01 * math.cos(2 * math.pi * k / n), 0.01 * math.sin(2 * math.pi * k / n), HIDDEN[2]] for k in range(n)]
    return [HIDDEN] + rim

FAN_NODE = """DEF SENSOR_FAN_POSE Pose {{ children [ DEF SENSOR_FAN Shape {{
  appearance PBRAppearance {{ baseColor 0.1 0.8 0.9 emissiveColor 0.1 0.8 0.9 emissiveIntensity 0.3 transparency 0.72 roughness 1 metalness 0 }}
  geometry IndexedFaceSet {{
    coord DEF SENSOR_FAN_POINTS Coordinate {{ point [ {points} ] }}
    coordIndex [ {faces} ]
    creaseAngle 0
    ccw FALSE
  }}
}} ] }}"""                  # in a Pose: a bare Shape cannot be hidden from cameras
RING_NODE = """DEF SENSOR_PERSON{i} Pose {{
  translation 0 0 -1000
  children [ Shape {{
    appearance PBRAppearance {{ baseColor 0.1 0.9 0.2 emissiveColor 0.1 0.9 0.2 emissiveIntensity 0.6 transparency 0.35 roughness 1 metalness 0 }}
    geometry Cylinder {{ height 0.01 radius 0.5 }} }} ]
}}"""


class SensorView:
    """The fan and the rings. `update()` redraws them from the chair's lidar and the people it can
    see; `show(False)` hides them."""

    def __init__(self, scene):
        from webots.people import line_of_sight
        self.scene, self.line_of_sight = scene, line_of_sight
        sup = scene.sup
        n = RAYS_SHOWN
        if sup.getFromDef("SENSOR_FAN_POSE") is None:
            children = sup.getRoot().getField("children")
            points = " ".join(" ".join(f"{v:g}" for v in point) for point in hidden_fan(n))
            faces = " ".join(f"0 {i + 1} {(i + 1) % n + 1} -1" for i in range(n))
            children.importMFNodeFromString(-1, FAN_NODE.format(points=points, faces=faces))
            for i in range(PEOPLE_RINGS):
                children.importMFNodeFromString(-1, RING_NODE.format(i=i))
        for name in ["SENSOR_FAN_POSE"] + [f"SENSOR_PERSON{i}" for i in range(PEOPLE_RINGS)]:
            scene.only_on_screen(sup.getFromDef(name))   # the chair's view only: not on the phones
        self.points = sup.getFromDef("SENSOR_FAN_POINTS").getField("point")
        self.rings = [sup.getFromDef(f"SENSOR_PERSON{i}").getField("translation") for i in range(PEOPLE_RINGS)]
        self.on = True
        self.visible = True

    def show(self, on: bool) -> None:
        self.on = on
        if not on:
            self._hide()

    def _hide(self) -> None:
        for i, point in enumerate(hidden_fan(RAYS_SHOWN)):
            self.points.setMFVec3f(i, point)
        for ring in self.rings:
            ring.setSFVec3f(HIDDEN)

    def update(self) -> None:
        if not self.on:
            return
        scene = self.scene
        r = scene.robot
        image = scene.lidar.getRangeImage()
        n = len(image)
        reach = scene.lidar.getMaxRange()
        self.points.setMFVec3f(0, [r.x, r.y, FAN_Z])
        for k in range(RAYS_SHOWN):
            i = k * n // RAYS_SHOWN
            d = image[i]
            d = reach if not math.isfinite(d) else min(d, reach)
            # Webots orders a 360 degree lidar's rays from straight behind, turning clockwise.
            angle = r.orientation + math.pi - 2 * math.pi * i / n
            self.points.setMFVec3f(k + 1, [r.x + d * math.cos(angle), r.y + d * math.sin(angle), FAN_Z])
        seen = [p for p in scene.crowd.everyone
                if self.line_of_sight(scene.plan, (r.x, r.y), (p.x, p.y))
                and math.dist((r.x, r.y), (p.x, p.y)) <= reach][:PEOPLE_RINGS]
        for i, ring in enumerate(self.rings):
            ring.setSFVec3f([seen[i].x, seen[i].y, 0.015] if i < len(seen) else HIDDEN)
