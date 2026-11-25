"""While the rider drives (O in run.py), the chair watches for people in its way: a warning at the
top of the screen and a red ring under the person, and it slows down for them (run.py --no-brake:
the warning only). It slows down only going forward, never turning, so the rider can always turn
away. The people it watches for are the ones its sensors can see (the green rings of the sensor
view: in line of sight, within the lidar's reach), not everyone in the simulation.

It also watches for walls and anything else its lidar sees straight ahead (furniture, a trolley),
so the rider cannot drive it into them: an OBSTACLE warning, slower from WALL_SLOW_FROM_M and no
forward speed at WALL_STOP_M. Walls can come nearer than people, so a doorway the chair fits
through stays open.

Speed allowed going forward, by the gap between the chair and the person in its way (edge to edge,
the same distance that counts as a hit at 0):

    gap 1.2 m or more   full speed
    1.2 m to 0.3 m      less and less
    0.3 m or less       none (turn away)
"""

from __future__ import annotations

import math

import numpy as np

WARN_M = 1.5                # someone this near in the chair's way: the warning and the red ring
SLOW_FROM_M = 1.2           # ... and nearer than this, slower
STOP_M = 0.3                # ... and this near, no forward speed at all
PATH_MARGIN_M = 0.15        # in the chair's way: within its width, plus this, straight ahead
WALL_WARN_M = 0.8           # a wall or anything else the lidar sees this near, straight ahead: the warning
WALL_SLOW_FROM_M = 0.8      # ... and nearer than this, slower
WALL_STOP_M = 0.12          # ... and this near, no forward speed at all
WALL_GRAZE_M = 0.02         # something this much inside the chair's width, to one side, still counts
RING_NODE = """DEF GUARD_RING Pose {
  translation 0 0 -1000
  children [ Shape {
    appearance PBRAppearance { baseColor 0.9 0.1 0.1 emissiveColor 0.9 0.1 0.1 emissiveIntensity 0.8 transparency 0.25 roughness 1 metalness 0 }
    geometry Cylinder { height 0.012 radius 0.55 } } ]
}"""
HIDDEN = [0.0, 0.0, -1000.0]


def in_the_way(x, y, heading, people, chair_radius: float):
    """The nearest person straight ahead of a chair at (x, y) facing `heading`: (gap in metres, person),
    or (inf, None). `people` have x, y and width (an adult 0.72 m)."""
    from webots.bridge import PERSON_DIAMETER_M
    best = (math.inf, None)
    c, s = math.cos(heading), math.sin(heading)
    for p in people:
        dx, dy = p.x - x, p.y - y
        ahead, aside = dx * c + dy * s, -dx * s + dy * c
        width = getattr(p, "width", PERSON_DIAMETER_M)
        if ahead <= 0 or abs(aside) > chair_radius + width / 2 + PATH_MARGIN_M:
            continue
        gap = math.hypot(dx, dy) - chair_radius - PERSON_DIAMETER_M / 2 - (width - PERSON_DIAMETER_M) / 2
        if gap < best[0]:
            best = (gap, p)
    return best


def allowed(gap: float) -> float:
    """The share of full forward speed allowed with someone `gap` metres ahead."""
    return min(1.0, max(0.0, (gap - STOP_M) / (SLOW_FROM_M - STOP_M)))


def wall_ahead(ranges, chair_radius: float) -> float:
    """The gap in metres between the front of the chair and the nearest thing its lidar sees straight
    ahead (inf if nothing). `ranges` is the lidar's range image: a 360 degree lidar at the chair's
    centre, its rays from straight behind, turning clockwise (Webots' order, as in bridge.py)."""
    r = np.asarray(ranges, dtype=float)
    n = len(r)
    if n == 0:
        return math.inf
    angle = math.pi - 2 * math.pi * np.arange(n) / n        # anticlockwise from straight ahead
    seen = np.isfinite(r) & (r > 0)
    r = np.where(seen, r, 0.0)
    x, y = r * np.cos(angle), r * np.sin(angle)
    ahead = seen & (x > 0) & (np.abs(y) < chair_radius - WALL_GRAZE_M)
    if not ahead.any():
        return math.inf
    front = np.sqrt(np.maximum(chair_radius ** 2 - y[ahead] ** 2, 0.0))   # the chair's edge at that side offset
    return float(np.min(x[ahead] - front))


def wall_allowed(gap: float) -> float:
    """The share of full forward speed allowed with a wall or obstacle `gap` metres ahead."""
    return min(1.0, max(0.0, (gap - WALL_STOP_M) / (WALL_SLOW_FROM_M - WALL_STOP_M)))


class Guard:
    """`check(command)` each step the rider drives: the command, slowed down if someone is in the
    way. `warning` is the text for the screen ("" when nobody is), `hits` counts the people hit
    while the rider drove. `off()` when the chair drives itself again."""

    def __init__(self, scene, brake: bool = True):
        from webots.people import line_of_sight
        self.scene, self.brake, self.line_of_sight = scene, brake, line_of_sight
        sup = scene.sup
        if sup.getFromDef("GUARD_RING") is None:
            sup.getRoot().getField("children").importMFNodeFromString(-1, RING_NODE)
        node = sup.getFromDef("GUARD_RING")
        scene.only_on_screen(node)
        self.ring = node.getField("translation")
        self.warning, self.hits, self.touching, self.ring_at = "", 0, False, None
        self.wall_hits, self.touching_wall = 0, False

    def seen(self) -> list:
        sc = self.scene
        r = sc.robot
        reach = sc.lidar.getMaxRange()
        return [p for p in sc.crowd.everyone
                if math.dist((r.x, r.y), (p.x, p.y)) <= reach and self.line_of_sight(sc.plan, (r.x, r.y), (p.x, p.y))]

    def check(self, command, info=None):
        """The rider's command, slowed down for whoever or whatever is in the way. `info` (the last
        step's, from the scene) counts a hit when the chair has just touched someone or a wall."""
        from webots.bridge import CHAIR_RADIUS_M
        if info is not None:
            touching = bool(info.get("COLLISION_HUMAN"))
            self.hits += touching and not self.touching
            self.touching = touching
            touching_wall = bool(info.get("COLLISION")) and not touching
            self.wall_hits += touching_wall and not self.touching_wall
            self.touching_wall = touching_wall
        r = self.scene.robot
        gap, person = in_the_way(r.x, r.y, r.orientation, self.seen(), CHAIR_RADIUS_M)
        share = allowed(gap)
        wall = self.wall_gap()
        wall_share = wall_allowed(wall)
        if gap < WARN_M:
            what = "stopped: turn away" if self.brake and share == 0.0 else "slowing down" if self.brake and share < 1.0 \
                else "careful"
            self.warning = f"PERSON {max(gap, 0.0):.1f} m AHEAD: {what}"
            self._ring((person.x, person.y))
        elif wall < WALL_WARN_M:
            what = "stopped: turn away" if self.brake and wall_share == 0.0 else "slowing down" \
                if self.brake and wall_share < 1.0 else "careful"
            self.warning = f"OBSTACLE {max(wall, 0.0):.1f} m AHEAD: {what}"
            self._ring(None)
        else:
            self.warning = ""
            self._ring(None)
        share = min(share, wall_share)
        if self.brake and command[0] > share:
            command = command.copy()
            command[0] = share
        return command

    def wall_gap(self) -> float:
        """How far ahead the nearest wall or obstacle is, from the chair's lidar (inf without one)."""
        from webots.bridge import CHAIR_RADIUS_M
        lidar = getattr(self.scene, "lidar", None)
        image = lidar.getRangeImage() if lidar is not None and hasattr(lidar, "getRangeImage") else None
        return wall_ahead(image, CHAIR_RADIUS_M) if image is not None and len(image) else math.inf

    def off(self) -> None:
        self.warning, self.touching, self.touching_wall = "", False, False
        self._ring(None)

    def _ring(self, xy) -> None:
        spot = None if xy is None else (round(xy[0], 2), round(xy[1], 2))
        if spot != self.ring_at:                   # only when it moves: one change for Webots, not every step
            self.ring_at = spot
            self.ring.setSFVec3f(HIDDEN if spot is None else [spot[0], spot[1], 0.02])
