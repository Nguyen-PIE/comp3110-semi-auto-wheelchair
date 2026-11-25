"""Load a building floor plan and work out where the chair and people can go.

A floor plan is a ROS-style map: an image (black = wall, white = open floor,
grey = unknown or outside) and a YAML file giving its scale and origin, e.g.

    image: hospital.png
    resolution: 0.05          # metres per pixel
    origin: [-49.6, -4.7, 0]  # map position of the bottom-left pixel
    negate: 0
    occupied_thresh: 0.65
    free_thresh: 0.196

wheelchair-navi ships three of these (hospital, office, warehouse), which are
maps of the same buildings Isaac Sim provides in 3D. scripts/import_floorplan.py
turns any floor-plan picture into one.

Coordinates: SocNavGym needs the map centred on (0, 0), so the plan is cropped
to the building and shifted. `to_map_frame()` undoes the shift, so positions
can be matched with the original map (and Isaac Sim) later.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

ROUTE_CELL_M = 0.25    # route planning grid size
CLEARANCE_M = 0.45     # the chair's centre must stay this far from walls (radius 0.4 m + 5 cm)
PERSON_CLEARANCE_M = 0.4   # a walking person's centre (shoulders 0.72 m wide)
COMFORT_M = 1.0        # routes prefer to keep at least this far from walls
WALL_PENALTY = 4.0     # how strongly routes avoid passing closer than COMFORT_M
SPAWN_ROOM_M = 0.75    # spawn points and goals need this much room around them

FREE, WALL, UNKNOWN = 0, 1, 2


@dataclass
class FloorPlan:
    name: str
    resolution: float          # metres per pixel of `cells` and `clearance`
    width_m: float
    height_m: float
    offset: tuple[float, float]  # add to (x, y) to get back to the map's own frame
    cells: np.ndarray          # per pixel: FREE, WALL or UNKNOWN
    clearance: np.ndarray      # per pixel: metres to the nearest wall or unknown pixel
    grad_x: np.ndarray         # direction away from the nearest wall, per pixel
    grad_y: np.ndarray
    blocked: np.ndarray        # route grid: True where the chair's centre cannot be
    step_cost: np.ndarray      # route grid: cost multiplier, higher close to walls
    roomy: np.ndarray          # route grid: True where there is room to pass easily
    spawn_cells: np.ndarray    # route-grid (row, col) pairs that are reachable and roomy
    cell_clearance: np.ndarray  # route grid: clearance at each cell's centre

    # ---------- coordinates ----------
    def to_map_frame(self, x: float, y: float) -> tuple[float, float]:
        return x + self.offset[0], y + self.offset[1]

    def _pixel(self, x: float, y: float) -> tuple[int, int]:
        rows, cols = self.cells.shape
        col = int((x + self.width_m / 2) / self.resolution)
        row = int((self.height_m / 2 - y) / self.resolution)
        return min(max(row, 0), rows - 1), min(max(col, 0), cols - 1)

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        rows, cols = self.blocked.shape
        col = int((x + self.width_m / 2) / ROUTE_CELL_M)
        row = int((self.height_m / 2 - y) / ROUTE_CELL_M)
        return min(max(row, 0), rows - 1), min(max(col, 0), cols - 1)

    def centre_of(self, row: int, col: int) -> tuple[float, float]:
        return (col + 0.5) * ROUTE_CELL_M - self.width_m / 2, self.height_m / 2 - (row + 0.5) * ROUTE_CELL_M

    # ---------- walls ----------
    def clearance_at(self, x: float, y: float) -> float:
        """Distance in metres from (x, y) to the nearest wall (0 inside a wall or outside the map)."""
        if abs(x) >= self.width_m / 2 or abs(y) >= self.height_m / 2:
            return 0.0
        return float(self.clearance[self._pixel(x, y)])

    def away_from_wall(self, x: float, y: float) -> tuple[float, float]:
        """Unit direction away from the nearest wall at (x, y), or (0, 0) if there is none nearby."""
        r, c = self._pixel(x, y)
        gx, gy = float(self.grad_x[r, c]), float(self.grad_y[r, c])
        norm = math.hypot(gx, gy)
        return (gx / norm, gy / norm) if norm > 1e-6 else (0.0, 0.0)

    def push_out(self, x: float, y: float, radius: float) -> tuple[float, float]:
        """Move a circle of `radius` at (x, y) out of any wall it overlaps."""
        r, c = self._pixel(x, y)
        gap = radius - float(self.clearance[r, c])
        if gap <= 0:
            return x, y
        gx, gy = float(self.grad_x[r, c]), float(self.grad_y[r, c])
        norm = math.hypot(gx, gy)
        if norm < 1e-6:
            return x, y
        return x + gx / norm * (gap + 0.01), y + gy / norm * (gap + 0.01)

    # ---------- sampling ----------
    def random_point(self, rng, cells: np.ndarray | None = None) -> tuple[float, float]:
        """A random reachable spot with room around it. `rng` is Python's random module or a Random."""
        cells = self.spawn_cells if cells is None or len(cells) == 0 else cells
        row, col = cells[rng.randrange(len(cells))]
        x, y = self.centre_of(int(row), int(col))
        jitter = ROUTE_CELL_M / 2
        return x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)

    def cells_near(self, points: list[tuple[float, float]], radius: float) -> np.ndarray:
        """Spawn cells within `radius` metres of the polyline through `points`."""
        centres = np.stack([(self.spawn_cells[:, 1] + 0.5) * ROUTE_CELL_M - self.width_m / 2,
                            self.height_m / 2 - (self.spawn_cells[:, 0] + 0.5) * ROUTE_CELL_M], axis=1)
        best = np.full(len(centres), np.inf)
        pts = np.asarray(points, dtype=float)
        if len(pts) == 1:
            pts = np.vstack([pts, pts])
        for a, b in zip(pts[:-1], pts[1:]):
            ab = b - a
            # Written out rather than `(centres - a) @ ab`: on Apple Silicon, NumPy's matmul goes through
            # Apple's Accelerate library, which prints false "divide by zero / overflow in matmul" warnings.
            d = centres - a
            t = np.clip((d[:, 0] * ab[0] + d[:, 1] * ab[1]) / max(float(ab[0] ** 2 + ab[1] ** 2), 1e-9), 0.0, 1.0)
            nearest = a + t[:, None] * ab
            best = np.minimum(best, np.linalg.norm(centres - nearest, axis=1))
        return self.spawn_cells[best <= radius]

    # ---------- routes ----------
    def _nearest_open(self, cell, blocked=None):
        """The closest open cell, in case a point sits just inside the clearance margin."""
        blocked = self.blocked if blocked is None else blocked
        r0, c0 = cell
        if not blocked[r0, c0]:
            return cell
        rows, cols = blocked.shape
        for radius in range(1, 12):
            for r in range(r0 - radius, r0 + radius + 1):
                for c in (c0 - radius, c0 + radius) if abs(r - r0) != radius else range(c0 - radius, c0 + radius + 1):
                    if 0 <= r < rows and 0 <= c < cols and not blocked[r, c]:
                        return r, c
        return None

    def _line_is_roomy(self, a, b, roomy=None) -> bool:
        roomy = self.roomy if roomy is None else roomy
        (r0, c0), (r1, c1) = a, b
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        for i in range(steps + 1):
            if not roomy[round(r0 + (r1 - r0) * i / steps), round(c0 + (c1 - c0) * i / steps)]:
                return False
        return True

    def _line_keeps(self, a, b, clearance: float, blocked) -> bool:
        """Whether the straight line between two route cells stays open and at least `clearance`
        metres from walls."""
        (r0, c0), (r1, c1) = a, b
        steps = max(abs(r1 - r0), abs(c1 - c0), 1) * 2
        t = np.linspace(0.0, 1.0, steps + 1)
        rows = np.round(r0 + (r1 - r0) * t).astype(int)
        cols = np.round(c0 + (c1 - c0) * t).astype(int)
        return bool(not blocked[rows, cols].any() and (self.cell_clearance[rows, cols] >= clearance).all())

    def _grids(self, for_people: bool):
        """The blocked and roomy route grids for the chair, or for a person walking (who needs
        less room: PERSON_CLEARANCE_M from walls instead of CLEARANCE_M)."""
        if not for_people:
            return self.blocked, self.roomy
        if getattr(self, "_people_blocked", None) is None:
            self._people_blocked = self.cell_clearance < PERSON_CLEARANCE_M
        return self._people_blocked, self.roomy

    def _with_obstacles(self, avoid, for_people: bool = False):
        """Copies of the blocked and roomy grids with circles (x, y, radius) added as obstacles."""
        base_blocked, base_roomy = self._grids(for_people)
        blocked, roomy = base_blocked.copy(), base_roomy.copy()
        rows, cols = blocked.shape
        margin = PERSON_CLEARANCE_M if for_people else CLEARANCE_M
        for x, y, radius in avoid:
            reach = radius + margin               # keep the walker's centre this far from the circle's centre
            r0, c0 = self.cell_of(x, y)
            k = int(math.ceil(reach / ROUTE_CELL_M)) + 1
            for r in range(max(r0 - k, 0), min(r0 + k + 1, rows)):
                for c in range(max(c0 - k, 0), min(c0 + k + 1, cols)):
                    cx, cy = self.centre_of(r, c)
                    if math.hypot(cx - x, cy - y) < reach:
                        blocked[r, c] = True
                    if math.hypot(cx - x, cy - y) < reach + 0.5:
                        roomy[r, c] = False
        return blocked, roomy

    def connects(self, start_xy, goal_xy) -> bool:
        """Whether the chair can drive from start to goal: both are on open floor (or within half a
        metre of it) in the reachable area, and there is a route between them."""
        for x, y in (start_xy, goal_xy):
            cell = self._nearest_open(self.cell_of(x, y))
            if cell is None or math.dist(self.centre_of(*cell), (x, y)) > 0.5:
                return False
        return self.route(start_xy, goal_xy) is not None

    def route(self, start_xy, goal_xy, avoid=(), for_people: bool = False) -> list[tuple[float, float]] | None:
        """Route from start to goal as a list of waypoints in metres (not including the start).

        Routes keep the chair's centre CLEARANCE_M from walls (a person's,
        PERSON_CLEARANCE_M, with for_people=True), prefer the middle of corridors
        and doorways, and are straightened where there is room. `avoid` adds
        temporary obstacles as circles (x, y, radius), such as people standing
        still or the chair. Returns None if the goal cannot be reached.
        """
        blocked, roomy = self._with_obstacles(avoid, for_people) if avoid else self._grids(for_people)
        start = self._nearest_open(self.cell_of(*start_xy), blocked)
        goal = self._nearest_open(self.cell_of(*goal_xy), blocked)
        if start is None or goal is None:
            return None
        rows, cols = blocked.shape
        step_cost = self.step_cost
        diag = math.sqrt(2)
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)]

        def h(cell):  # octile distance, never more than the true cost
            dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
            return (dr + dc) + (diag - 2) * min(dr, dc)

        came_from = {start: None}
        cost = {start: 0.0}
        frontier = [(h(start), start)]
        while frontier:
            _, cell = heapq.heappop(frontier)
            if cell == goal:
                break
            r, c = cell
            for dr, dc, step in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                    continue
                if dr and dc and (blocked[r, nc] or blocked[nr, c]):  # no cutting corners
                    continue
                new_cost = cost[cell] + step * step_cost[nr, nc]
                if new_cost < cost.get((nr, nc), math.inf):
                    cost[(nr, nc)] = new_cost
                    came_from[(nr, nc)] = cell
                    heapq.heappush(frontier, (new_cost + h((nr, nc)), (nr, nc)))
        if goal not in came_from:
            return None

        cells = []
        cell = goal
        while cell is not None:
            cells.append(cell)
            cell = came_from[cell]
        cells.reverse()

        # Straighten: skip waypoints where a straight line has plenty of room. Elsewhere, cut the
        # grid's zigzags where a straight line stays as far from walls as the grid route did
        # (zigzags made the chair swing left and right).
        kept = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._line_is_roomy(cells[i], cells[j], roomy):
                j -= 1
            if j == i + 1:
                need = min(COMFORT_M, float(self.cell_clearance[cells[i]]), float(self.cell_clearance[cells[j]]))
                while j + 1 < len(cells):
                    need = min(need, float(self.cell_clearance[cells[j + 1]]))
                    if not self._line_keeps(cells[i], cells[j + 1], need - 0.05, blocked):
                        break
                    j += 1
            kept.append(cells[j])
            i = j
        points = [self.centre_of(r, c) for r, c in kept[1:]]
        if not points:
            points = [tuple(goal_xy)]
        points[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        return points


def route_length(start_xy, waypoints) -> float:
    points = [tuple(start_xy)] + list(waypoints)
    return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))


def _downsample(mask: np.ndarray, k: int, rule: str) -> np.ndarray:
    """Shrink a boolean grid by k: a block is True if any (or all) of its pixels are.
    Pixels beyond the image edge count as False."""
    h, w = math.ceil(mask.shape[0] / k), math.ceil(mask.shape[1] / k)
    padded = np.zeros((h * k, w * k), dtype=bool)
    padded[: mask.shape[0], : mask.shape[1]] = mask
    blocks = padded.reshape(h, k, w, k)
    return blocks.any(axis=(1, 3)) if rule == "any" else blocks.all(axis=(1, 3))


@lru_cache(maxsize=8)
def load_floorplan(yaml_path: str | Path) -> FloorPlan:
    """Read a ROS map (YAML + image). Results are cached, so loading the same plan twice is free."""
    yaml_path = Path(yaml_path).resolve()
    meta = yaml.safe_load(yaml_path.read_text())
    image = cv2.imread(str(yaml_path.parent / meta["image"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read the floor plan image for {yaml_path}")
    res = float(meta["resolution"])
    p_occupied = image.astype(np.float32) / 255.0
    if not meta.get("negate", 0):
        p_occupied = 1.0 - p_occupied
    occupied = p_occupied > float(meta.get("occupied_thresh", 0.65))
    free = p_occupied < float(meta.get("free_thresh", 0.196))

    # Crop to the building (everything that is not unknown), plus a 0.5 m border.
    known = occupied | free
    known_rows = np.where(known.any(axis=1))[0]
    known_cols = np.where(known.any(axis=0))[0]
    if len(known_rows) == 0:
        raise ValueError(f"{yaml_path} has no walls or open floor")
    pad = int(round(0.5 / res))
    r0, r1 = max(known_rows[0] - pad, 0), min(known_rows[-1] + pad + 1, image.shape[0])
    c0, c1 = max(known_cols[0] - pad, 0), min(known_cols[-1] + pad + 1, image.shape[1])
    occupied, free = occupied[r0:r1, c0:c1], free[r0:r1, c0:c1]
    height_px, width_px = free.shape

    # Where the cropped centre sits in the map's own frame (origin is the bottom-left pixel).
    origin_x, origin_y = float(meta["origin"][0]), float(meta["origin"][1])
    centre_x = origin_x + (c0 + width_px / 2) * res
    centre_y = origin_y + (image.shape[0] - r0 - height_px / 2) * res
    return build_plan(yaml_path.stem, res, occupied, free, (centre_x, centre_y))


def room_plan(u, resolution: float = 0.05) -> FloorPlan:
    """A FloorPlan of one of SocNavGym's generated rooms (s1_corridor, s2_ward, ...), so the route
    follower and the wall sensor work there too. Walls, tables, chairs, laptops and plants
    become walls; people are not included (the people tracker handles them). Rooms change
    at every reset, so build a new plan after each one."""
    from socnavgym.envs.utils.utils import get_coordinates_of_rotated_rectangle

    width_m, height_m = u.MAP_X + 1.0, u.MAP_Y + 1.0       # 0.5 m border around the room
    rows, cols = int(math.ceil(height_m / resolution)), int(math.ceil(width_m / resolution))
    width_m, height_m = cols * resolution, rows * resolution

    def pixel(x, y):
        return [(x + width_m / 2) / resolution, (height_m / 2 - y) / resolution]

    occupied = np.zeros((rows, cols), dtype=np.uint8)
    for thing in list(u.walls) + list(u.tables) + list(u.chairs) + list(u.laptops):
        across = thing.thickness if thing.name == "wall" else thing.width
        corners = get_coordinates_of_rotated_rectangle(thing.x, thing.y, thing.orientation, thing.length, across)
        cv2.fillPoly(occupied, [np.round(np.array([pixel(x, y) for x, y in corners]) * 8).astype(np.int32)], 1,
                     shift=3)
    for plant in u.plants:
        centre = np.round(np.array(pixel(plant.x, plant.y)) * 8).astype(np.int32)
        cv2.circle(occupied, tuple(int(v) for v in centre), int(round(plant.radius / resolution * 8)), 1, -1, shift=3)
    inside = np.zeros((rows, cols), dtype=bool)          # the room's own rectangle
    x0, y0 = pixel(-u.MAP_X / 2, u.MAP_Y / 2)
    x1, y1 = pixel(u.MAP_X / 2, -u.MAP_Y / 2)
    inside[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))] = True
    occupied = occupied.astype(bool)
    return build_plan("room", resolution, occupied, inside & ~occupied, (0.0, 0.0))


def build_plan(name: str, res: float, occupied: np.ndarray, free: np.ndarray, offset: tuple[float, float],
               keep_xy: tuple[float, float] | None = None) -> FloorPlan:
    """Build a FloorPlan from occupancy grids (row 0 at the top) centred on (0, 0). Only the open
    area containing `keep_xy` (default: the biggest one) is reachable."""
    height_px, width_px = free.shape
    width_m, height_m = width_px * res, height_px * res
    cells = np.full(free.shape, UNKNOWN, dtype=np.uint8)
    cells[free] = FREE
    cells[occupied] = WALL

    # Distance from every open pixel to the nearest wall or unknown pixel.
    clearance = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * res
    gy_rows, gx_cols = np.gradient(clearance)
    grad_x, grad_y = gx_cols.astype(np.float32), (-gy_rows).astype(np.float32)

    # Route grid. A cell is open only if its whole area keeps CLEARANCE_M from walls.
    k = max(1, int(round(ROUTE_CELL_M / res)))
    route_open = _downsample(clearance >= CLEARANCE_M, k, "all")
    roomy = _downsample(clearance >= COMFORT_M, k, "all")
    centre_rows = np.minimum(np.arange(route_open.shape[0]) * k + k // 2, height_px - 1)
    centre_cols = np.minimum(np.arange(route_open.shape[1]) * k + k // 2, width_px - 1)
    cell_clearance = clearance[np.ix_(centre_rows, centre_cols)]
    closeness = np.clip((COMFORT_M - cell_clearance) / (COMFORT_M - CLEARANCE_M), 0.0, 1.0)
    step_cost = (1.0 + WALL_PENALTY * closeness).astype(np.float32)

    # Only the biggest connected open area (or the one containing keep_xy) counts; the rest is unreachable.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(route_open.astype(np.uint8), connectivity=8)
    if n <= 1:
        raise ValueError(f"No open floor wide enough for the chair in {name}")
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if keep_xy is not None:
        col = min(max(int((keep_xy[0] + width_m / 2) / ROUTE_CELL_M), 0), labels.shape[1] - 1)
        row = min(max(int((height_m / 2 - keep_xy[1]) / ROUTE_CELL_M), 0), labels.shape[0] - 1)
        r0, r1, c0, c1 = max(row - 4, 0), row + 5, max(col - 4, 0), col + 5
        near = labels[r0:r1, c0:c1]
        if (near > 0).any():
            keep = int(np.bincount(near[near > 0]).argmax())
    reachable = labels == keep
    spawn_ok = _downsample(clearance >= SPAWN_ROOM_M, k, "all")
    spawn_cells = np.argwhere(reachable & spawn_ok)
    blocked = ~reachable
    roomy &= reachable

    return FloorPlan(
        name=name,
        resolution=res,
        width_m=width_m,
        height_m=height_m,
        offset=offset,
        cells=cells,
        clearance=clearance.astype(np.float32),
        grad_x=grad_x,
        grad_y=grad_y,
        blocked=blocked,
        step_cost=step_cost,
        roomy=roomy,
        spawn_cells=spawn_cells,
        cell_clearance=cell_clearance.astype(np.float32),
    )
