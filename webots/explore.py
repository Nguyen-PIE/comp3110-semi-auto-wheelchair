"""Finding the way without the floor plan: the chair's own map from its lidar, and where to look next.

ChairMap: the chair starts knowing nothing of the building. Every few steps each lidar ray marks the
floor it crossed as open and the point it hit as solid (log-odds, so people who walk away are rubbed
out again by later rays). For planning, anything not seen yet counts as open: routes go through
unknown space and are planned again as walls appear. The map is the same grid as the building's
floor plan (5 cm cells, the same coordinates), but only its size comes from the plan, never what is
in it. Where the chair is comes from the simulator: a real chair would work it out from its own
sensors (SLAM), which this project does not do.

Explorer: the edge between open floor the chair has seen and space it has not ("frontiers"). It
picks the frontier with the best mix of near, big, and in the direction the signs point: a direction
sign's arrow for the place being looked for (or its department: a CT scanner is in Medical Imaging)
makes frontiers that way much cheaper.
"""

from __future__ import annotations

import difflib
import math
import re

import cv2
import numpy as np

from envs.floorplan import ROUTE_CELL_M, build_plan

HIT, MISS = 0.9, -0.4             # log-odds added for a ray's end point and for the floor it crossed
SOLID, OPEN = 0.8, -0.3           # above this is solid, below this has been seen open
CLAMP = (-2.0, 3.5)
LOOKED_M = 6.0                    # floor counts as explored only this near: the camera reads words up to about 5 m
FRONTIER_MIN_CELLS = 3            # a frontier must be at least this many route cells (0.75 m)
HINT_WEIGHT_M = 25.0              # a frontier straight along a sign's arrow is this much "nearer"
HINT_CONE = math.radians(60)
TRY_AGAIN_S = 60.0                # a frontier given up on is worth another try after this long
KEEP_M = 4.0                      # the frontier being driven to counts this much nearer than the others
LOOK_BONUS_M = 6.0                # a sign spotted but not read counts this much nearer than a frontier
NEAR_WEIGHT_M, NEAR_M = 15.0, 12.0   # rooms of the target's department read: up to this much nearer within NEAR_M

# Common knowledge about hospitals, as a visitor would have it: which department a room is in.
DEPARTMENT_OF = {
    "ct scan": "Medical Imaging", "x-ray": "Medical Imaging", "xray": "Medical Imaging", "mri": "Medical Imaging",
    "ultrasound": "Medical Imaging", "nuclear medicine": "Medical Imaging", "imaging reception": "Medical Imaging",
    "pathology collection": "Pathology", "pharmacy": "Pharmacy", "outpatient clinic": "Outpatient Clinics",
    "physiotherapy": "Allied Health", "occupational therapy": "Allied Health",
    "accessible toilet": "Toilets", "toilets": "Toilets",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def numbers(text: str) -> list[str]:
    """The numbers in a name with any letter stuck to them: "4A Room 3" -> ["4a", "3"]."""
    return re.findall(r"\d+[a-z]?", norm(text).replace(" ", ""))


def same_name(a: str, b: str) -> bool:
    """Whether a name read off a sign is the one asked for, allowing for a misread letter in its words,
    word by word ("Imaging Reception" is not "Main Reception"). Numbers must match exactly: "4B Room 1"
    is not "4B Room 5", and Ward 4A is not Ward 4B."""
    a, b = norm(a), norm(b)
    if numbers(a) != numbers(b):
        return False
    if a == b or a.replace(" ", "") == b.replace(" ", ""):
        return True
    wa, wb = a.split(), b.split()
    return len(wa) == len(wb) and all(_close_word(x, y) for x, y in zip(wa, wb))


def _close_word(x: str, y: str) -> bool:
    """The same word but for a misread letter or two ("Pharrnacy", "Toom" for "Room")."""
    if x == y:
        return True
    if difflib.SequenceMatcher(None, x, y).ratio() >= 0.8:
        return True
    if min(len(x), len(y)) < 4 or abs(len(x) - len(y)) > 1:
        return False
    if len(x) == len(y):
        return sum(p != q for p, q in zip(x, y)) <= 1
    short, long = sorted((x, y), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def departments_for(name: str, department: str | None = None) -> set[str]:
    """What direction signs might call the way to `name`: itself, and its department (given, as the
    building's directory lists it, or as a visitor would guess it)."""
    n = norm(name)
    out = {n}
    if department:
        out.add(norm(department))
    if n in DEPARTMENT_OF:
        out.add(norm(DEPARTMENT_OF[n]))
    m = re.match(r"(\d[a-z]) ", n)                 # "4A Room 3" is on Ward 4A
    if m:
        out.add(f"ward {m.group(1)}")
    if n.startswith("consult"):
        out.add("outpatient clinics")
    if n.startswith("loading dock"):
        out.add("loading docks")
    if n.startswith("aisle"):
        out.add("aisles")
    if n.startswith("meeting room"):
        out.add("meeting rooms")
    return out


class ChairMap:
    def __init__(self, plan):
        self.res = plan.resolution
        self.width_m, self.height_m = plan.width_m, plan.height_m
        self.offset = plan.offset
        self.shape = plan.cells.shape
        self.logodds = np.zeros(self.shape, np.float32)
        self.seen = np.zeros(self.shape, bool)
        self._plan, self._plan_at = None, -math.inf
        self.updates = 0

    def _cells(self, xs, ys):
        rows = ((self.height_m / 2 - ys) / self.res).astype(int)
        cols = ((xs + self.width_m / 2) / self.res).astype(int)
        ok = (rows >= 0) & (rows < self.shape[0]) & (cols >= 0) & (cols < self.shape[1])
        return rows[ok], cols[ok], ok

    def update(self, x: float, y: float, heading: float, ranges: np.ndarray, max_range: float) -> None:
        """Add one lidar scan taken at (x, y, heading). `ranges` in Webots' order: from straight behind,
        turning clockwise. Walls and open floor are mapped as far as the lidar reaches (for routes), but
        floor counts as explored only within LOOKED_M: from further off the camera cannot read signs, so
        a corridor mapped from its far end still has to be driven down."""
        n = len(ranges)
        angles = heading + math.pi - 2 * math.pi * np.arange(n) / n
        r = np.asarray(ranges, dtype=np.float32)
        hit = np.isfinite(r) & (r < max_range - 0.05)
        reach = np.where(hit, r, max_range)
        steps = np.arange(0.0, max_range, self.res / 1.5, dtype=np.float32)
        along = steps[None, :] < (reach[:, None] - self.res)
        px = x + np.cos(angles)[:, None] * steps[None, :]
        py = y + np.sin(angles)[:, None] * steps[None, :]
        rows, cols, ok = self._cells(px[along], py[along])
        free = np.zeros(self.shape, bool)
        free[rows, cols] = True
        hx, hy = x + np.cos(angles[hit]) * r[hit], y + np.sin(angles[hit]) * r[hit]
        hrows, hcols, _ = self._cells(hx, hy)
        solid = np.zeros(self.shape, bool)
        solid[hrows, hcols] = True
        free &= ~solid
        self.logodds[free] += MISS
        self.logodds[solid] += HIT
        np.clip(self.logodds, *CLAMP, out=self.logodds)
        k = int(LOOKED_M / self.res)
        r0, c0 = int((self.height_m / 2 - y) / self.res), int((x + self.width_m / 2) / self.res)
        ra, rb = max(r0 - k, 0), min(r0 + k + 1, self.shape[0])
        ca, cb = max(c0 - k, 0), min(c0 + k + 1, self.shape[1])
        if ra < rb and ca < cb:
            rr, cc = np.ogrid[ra:rb, ca:cb]
            near = (rr - r0) ** 2 + (cc - c0) ** 2 <= k * k
            self.seen[ra:rb, ca:cb] |= (free[ra:rb, ca:cb] | solid[ra:rb, ca:cb]) & near
        self.updates += 1

    def plan(self, chair_xy, now: float, every_s: float = 1.0):
        """A floor plan of what the chair knows (unknown counts as open), rebuilt at most every `every_s`."""
        if self._plan is None or now - self._plan_at >= every_s:
            solid = self.logodds > SOLID
            self._plan = build_plan("chair map", self.res, solid, ~solid, self.offset, keep_xy=chair_xy)
            self._plan_at = now
        return self._plan

    def known_open(self) -> np.ndarray:
        return self.seen & (self.logodds < OPEN)

    def frontiers(self, plan) -> list[tuple[float, float, int]]:
        """Open floor next to unseen space, where the chair could stand: [(x, y, size in route cells)]."""
        k = int(round(ROUTE_CELL_M / self.res))
        rows, cols = plan.blocked.shape
        def blocks(a, fill):                       # the pixels of each route cell (the grid rounds up)
            a = a[:rows * k, :cols * k]
            a = np.pad(a, ((0, rows * k - a.shape[0]), (0, cols * k - a.shape[1])), constant_values=fill)
            return a.reshape(rows, k, cols, k)
        known = blocks(self.known_open(), False).any(axis=(1, 3))
        unseen = blocks(~self.seen, True).all(axis=(1, 3))
        edge = known & cv2.dilate(unseen.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        edge &= ~plan.blocked
        n, labels, stats, _ = cv2.connectedComponentsWithStats(edge.astype(np.uint8), connectivity=8)
        out = []
        for i in range(1, n):
            size = int(stats[i, cv2.CC_STAT_AREA])
            if size < FRONTIER_MIN_CELLS:
                continue
            cells = np.argwhere(labels == i)
            mid = cells.mean(axis=0)
            r, c = cells[np.argmin(((cells - mid) ** 2).sum(axis=1))]
            out.append((*plan.centre_of(int(r), int(c)), size))
        return out

    def explored_share(self, plan_true) -> float:
        """How much of the building's open floor the chair has explored, from within LOOKED_M (for reports only)."""
        from envs.floorplan import FREE
        truth = plan_true.cells == FREE
        return float((self.seen & truth).sum() / max(1, truth.sum()))


def distances_from(plan, xy) -> np.ndarray:
    """Route-grid driving distance (m) from xy to every cell of `plan`."""
    from webots.signs import _distance_field
    return _distance_field(plan, [xy])


class Explorer:
    """Chooses where to look next."""

    def __init__(self):
        self.tried: list[tuple[float, float, float]] = []   # frontiers given up on (stuck): x, y, until when
        self.now = 0.0

    def give_up(self, xy, now: float) -> None:
        """Leave a spot the chair got stuck on its way to for TRY_AGAIN_S: people standing in a corridor
        are walls to the chair until they move, so it may be open later."""
        self.tried.append((xy[0], xy[1], now + TRY_AGAIN_S))

    def resting(self, xy) -> bool:
        return any(math.dist(xy, t[:2]) < 1.5 and self.now < t[2] for t in self.tried)

    def choose(self, plan, frontiers, here, hints=(), target: str | None = None, current=None,
               department: str | None = None, looks=(), near=(), wing: str | None = None):
        """Where to go next, as (x, y), or None when there is nothing left to explore or read.
        frontiers   [(x, y, size in route cells)]: the edge of what the chair has explored
        hints       [(name on a direction sign, sign x, sign y, the way it points)]
        current     the spot being driven to, kept unless another is better by KEEP_M (no dithering)
        department  the target's (a directory entry, or as a visitor would guess it)
        wing        the target's wing, from a directory: signs pointing to it count as pointing to the target
        looks       [(x, y) or (x, y, bonus)]: spots in front of signs spotted (by their tags) but not
                    read yet, LOOK_BONUS_M nearer than they are (or `bonus`)
        near        [(x, y)]: places read in the target's department (its rooms are together)"""
        if not frontiers and not looks:
            return None
        dist = distances_from(plan, here)
        wanted = (departments_for(target, department) if target else set()) | ({norm(wing)} if wing else set())
        useful = [(sx, sy, way) for name, sx, sy, way in hints if any(same_name(name, w) for w in wanted)]
        best, best_cost = None, math.inf
        candidates = [(x, y, size, None) for x, y, size in frontiers] + \
            [(lk[0], lk[1], 1, lk[2] if len(lk) > 2 else LOOK_BONUS_M) for lk in looks]
        for x, y, size, look in candidates:
            if self.resting((x, y)):
                continue
            cell = plan._nearest_open(plan.cell_of(x, y))
            if cell is None or not np.isfinite(dist[cell]):
                continue
            cost = dist[cell] - (look if look is not None else 2.0 * math.log(size))
            for sx, sy, way in useful:            # along an arrow for what we are looking for
                a = math.atan2(y - sy, x - sx)
                off = abs(math.atan2(math.sin(a - way), math.cos(a - way)))
                if off < HINT_CONE:
                    cost -= HINT_WEIGHT_M * math.cos(off * (math.pi / 2) / HINT_CONE)
                elif off > math.pi - HINT_CONE / 2 and math.dist((x, y), (sx, sy)) > 3.0:
                    cost += HINT_WEIGHT_M / 2       # straight back the way the sign says not to go
            if near:                              # close to rooms of the same department
                d = min(math.dist((x, y), p) for p in near)
                cost -= NEAR_WEIGHT_M * max(0.0, 1.0 - d / NEAR_M)
            if current is not None and math.dist((x, y), current) < 1.0:
                cost -= KEEP_M
                x, y = current                       # the same spot, a little changed by the mapping
            if cost < best_cost:
                best, best_cost = (x, y), cost
        return best
