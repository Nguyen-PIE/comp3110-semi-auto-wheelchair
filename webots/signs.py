"""Signs: named rooms and direction signs, as a real building has, for the chair to read with its camera.

    python webots/open_world.py fp_hospital          # the chair finds rooms by their signs (open_world.py)
    python webots/run.py fp_hospital --signs         # the signs in an ordinary trip, to look at
    python webots/signs.py fp_hospital               # list a building's signs

Four kinds, sized and placed as in Australian hospitals (see the README for the sources):
    door plate      beside the door on the handle side, 1.45 m up, 0.50 x 0.18 m: the room's name
    blade sign      sticking out from the wall above the door, 2.15 m up, 0.72 x 0.24 m, the name on both
                    sides, so it can be read from down the corridor
    direction sign  hanging at a corridor junction, 2.25 m up, 1.10 x 0.42 m: up to four destinations
                    with arrows, one face for each corridor that leads to the junction. In a building with
                    wings the other wings come first ("East Wing"), then the departments of this one
    directory       on the wall at the main reception and at junctions further off (so one is never far),
                    1.40 m up, 1.60 x 1.00 m: "You are here" (the wing) and the building's departments and
                    public places, each with its wing, as hospital directories list them
Each department has a colour (wards navy, medical imaging teal, outpatients purple, emergency red,
amenities green, staff only grey). Rooms with no door on the floor plan (the hospital's closed rooms,
the corridor's walls) get a door drawn on the wall.

Every sign face also carries a small square code (an ArUco tag, which OpenCV reads), as some
hospitals put QR codes on their signs. The tag says where the sign is and which way it faces, so
the chair can put it on its map (vision.py); the name itself is read from the text. Tag numbers
come in three blocks, one per kind of sign, so a tag also tells its sign's size and layout:
plates 0 to 59, blade faces 60 to 189, direction faces 190 to 234, directories 235 to 249.

Names: the hospital, the office, the corridor and the house are named by hand (below), every room by
a spot in the corridor in front of its door. Warehouses and the depot get names from lists, along
their outer walls and on the ends of their shelving. The hospital has three wings (West Wing, Central
Wing, East Wing) and the office three (North Wing, Central Wing, South Wing); a department with several rooms (a
ward) also gets a sign of its own at its entrance. The house has room signs only (as dementia-friendly
homes use) and no directory. Every building's variants (_v1, _v2, _v3, _test) have the same names: they
are the same building with other furniture.

The signs are added to the world while it runs (like landmarks.py), never saved in the .wbt, and
have no physics: the lidar does not see them and they change nothing about how the chair or the
people move, so training and testing are unchanged. Their pictures are drawn once into
webots/worlds/signs/<world>/ (not kept in Git), with a list of every sign face (signs.json) that
evaluation scripts use as the true answer.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

WEBOTS_DIR = Path(__file__).resolve().parent
ROOT = WEBOTS_DIR.parent
sys.path.insert(0, str(ROOT))

from envs.floorplan import FREE, ROUTE_CELL_M, UNKNOWN, WALL, load_floorplan  # noqa: E402

TEXTURES = WEBOTS_DIR / "worlds" / "signs"
ARUCO = cv2.aruco.DICT_4X4_250
PIXELS_PER_M = 800          # texture resolution

# kind: width, height, centre height (m), tag side (m, with its black border), first tag number, count
KINDS = {
    "plate": dict(w=0.50, h=0.18, z=1.45, tag=0.13, first=0, count=60),
    "blade": dict(w=0.72, h=0.24, z=2.15, tag=0.17, first=60, count=130),
    "direction": dict(w=1.10, h=0.42, z=2.25, tag=0.20, first=190, count=45),
    "directory": dict(w=1.60, h=1.00, z=1.40, tag=0.15, first=235, count=15),
}
TAG_MARGIN = 0.025          # white around the tag
SIGN_GAP_M = 1.0            # one door's plate (or blade) this far from the next door's
MIN_SIGN_GAP_M = 0.8        # ... or at least this far where there is no room
WALL_OFFSET_M = 0.10        # a plate's face is this far in front of the plan's wall pixels (walls are 10 cm blocks)
DOOR_W, DOOR_H = 0.95, 2.05
MAX_ROWS = 4                # destinations on one direction sign
DIRECTORY_ROWS = 9          # a directory lists two columns of this many
DIRECTORY_EVERY_M = 22.0    # a directory within this walk of every junction (and one at the main reception)
MAX_DIRECTORIES = 6

COLOURS = {                 # department colour (red, green, blue 0-255) for the sign background
    "ward": (31, 58, 95),
    "imaging": (0, 112, 122),
    "outpatients": (91, 44, 131),
    "emergency": (179, 0, 27),
    "allied": (138, 75, 8),
    "amenities": (46, 107, 48),
    "staff": (74, 74, 74),
    "office": (31, 58, 95),
    "warehouse": (24, 70, 120),
    "safety": (46, 107, 48),
    "home": (60, 90, 120),
}


@dataclass
class Place:
    """A named room or area. `x, y` is a spot in the corridor in front of its door (or in the open
    area), and `wall` the side its wall is on from there ("N", "S", "E" or "W"); without `wall`, the
    nearest wall. `group` is what direction signs call it (a department, or the place itself).
    area: an open area (a cafe, a waiting area): signs on the wall, no door.
    door: a room that is not on the floor plan: a door is drawn on the wall."""
    name: str
    group: str
    colour: str
    x: float
    y: float
    wall: str | None = None
    staff: bool = False
    area: bool = False
    door: bool = False
    entrance: bool = False      # a department's own sign at its entrance (made by make_layout)


@dataclass
class Face:
    """One readable face of a sign: what the chair's camera sees, and the true answer for evaluation."""
    tag: int
    kind: str                     # plate, blade or direction
    x: float                      # centre of the face, metres
    y: float
    z: float
    yaw: float                    # the way the face looks (its normal), radians
    place: str = ""               # plate and blade: the room's name
    group: str = ""
    colour: str = ""
    staff: bool = False
    rows: list = field(default_factory=list)   # direction: [(group, arrow)], arrow "left", "ahead" or "right"
                                               # directory: [(place or department, wing)]
    stop: tuple = (0.0, 0.0)      # where the chair stops to be at this place (plate, blade), or to read it (directory)
    here: str = ""                # directory: the wing it is in ("You are here")

    @property
    def text(self) -> str:
        """The words on the face, as a reader should get them."""
        if self.kind == "direction":
            return " | ".join(f"{g} {a}" for g, a in self.rows)
        if self.kind == "directory":
            return f"You are here: {self.here} | " + " | ".join(f"{n} {w}".strip() for n, w in self.rows)
        return self.place


# ---------------------------------------------------------------------------------- named places
def _hospital() -> list[Place]:
    """The hospital (Isaac Sim's hospital floor plan): two ward wings, Medical Imaging and Outpatients
    along the middle, the entrance hall and amenities. Spots are in the corridor in front of each door."""
    P = Place
    left, right = -30.4, -28.2               # the left wing corridor's two sides
    left_b, right_b = 28.9, 31.2             # the right wing's
    upper, lower = -4.3, -6.6                # the middle corridor's two sides
    return [
        # Ward 4A, the left wing: rooms on both sides of its corridor
        P("4A Room 1", "Ward 4A", "ward", left, 14.9, "W"), P("Clean Utility", "Ward 4A", "staff", left, 11.1, "W", staff=True),
        P("4A Room 2", "Ward 4A", "ward", left, 7.4, "W"), P("4A Room 3", "Ward 4A", "ward", left, 0.1, "W"),
        P("Dirty Utility", "Ward 4A", "staff", left, -3.6, "W", staff=True), P("4A Room 4", "Ward 4A", "ward", left, -7.4, "W"),
        P("Treatment Room", "Ward 4A", "ward", right, 14.9, "E"), P("4A Room 5", "Ward 4A", "ward", right, 11.1, "E"),
        P("Medication Room", "Ward 4A", "staff", right, 7.4, "E", staff=True),
        P("Staff Room", "Ward 4A", "staff", right, 3.6, "E", staff=True), P("4A Room 6", "Ward 4A", "ward", right, -0.1, "E"),
        P("Nurses' Station 4A", "Ward 4A", "ward", -36.4, 3.8, "W", area=True),
        P("Emergency", "Emergency", "emergency", -33.0, -13.8, "S", area=True),
        # Medical Imaging: the rooms above the middle corridor
        P("X-Ray", "Medical Imaging", "imaging", -10.8, upper, "N"), P("Ultrasound", "Medical Imaging", "imaging", -7.3, upper, "N"),
        P("Imaging Reception", "Medical Imaging", "imaging", 3.7, upper, "N"),
        P("CT Scan", "Medical Imaging", "imaging", 6.6, upper, "N"), P("MRI", "Medical Imaging", "imaging", 10.1, upper, "N"),
        P("Nuclear Medicine", "Medical Imaging", "imaging", 13.6, upper, "N"),
        # Outpatients: the rooms below it
        P("Pharmacy", "Pharmacy", "outpatients", -8.6, lower, "S"),
        P("Pathology Collection", "Pathology", "outpatients", 3.6, lower, "S"),
        P("Outpatient Clinic", "Outpatient Clinics", "outpatients", 10.6, lower, "S"),
        P("Consult Room 1", "Outpatient Clinics", "outpatients", 14.9, -10.4, "W"),
        # Ward 4B, the right wing
        P("4B Room 1", "Ward 4B", "ward", left_b, 10.9, "W"), P("Staff Room 4B", "Ward 4B", "staff", left_b, 7.15, "W", staff=True),
        P("4B Room 2", "Ward 4B", "ward", left_b, 3.4, "W"), P("4B Room 3", "Ward 4B", "ward", left_b, -0.35, "W"),
        P("Discharge Lounge", "Ward 4B", "ward", left_b, -4.1, "W"),
        P("Physiotherapy", "Allied Health", "allied", right_b, 10.65, "E"),
        P("Occupational Therapy", "Allied Health", "allied", right_b, 6.9, "E"),
        P("4B Room 4", "Ward 4B", "ward", right_b, 3.15, "E"), P("Dirty Utility 4B", "Ward 4B", "staff", right_b, -0.6, "E", staff=True),
        P("4B Room 5", "Ward 4B", "ward", right_b, -4.1, "E"), P("Clean Utility 4B", "Ward 4B", "staff", right_b, -7.85, "E", staff=True),
        P("Linen Store", "Ward 4B", "staff", right_b, -11.6, "E", staff=True),
        # Amenities
        P("Toilets", "Toilets", "amenities", 25.8, -14.6, "W"), P("Accessible Toilet", "Toilets", "amenities", 25.8, -17.6, "W"),
        P("Cafe", "Cafe", "amenities", -20.0, 6.8, "N", area=True),
        P("Main Reception", "Main Reception", "amenities", -20.8, 2.0, "W", area=True),
        P("Waiting Area", "Waiting Area", "amenities", 4.6, -13.6, area=True),
        P("Lifts", "Lifts", "amenities", 22.5, 17.5, "W", area=True),
    ]


def _corridor() -> list[Place]:
    """An outpatients corridor: doors along both long walls, Medical Imaging through the far end."""
    P = Place
    return [
        P("Consult 1", "Outpatient Clinics", "outpatients", -7.0, 1.6, door=True),
        P("Consult 2", "Outpatient Clinics", "outpatients", -1.0, 1.6, door=True),
        P("Treatment Room", "Outpatient Clinics", "outpatients", 5.0, 1.6, door=True),
        P("Consult 3", "Outpatient Clinics", "outpatients", -5.0, -1.6, door=True),
        P("Consult 4", "Outpatient Clinics", "outpatients", 1.0, -1.6, door=True),
        P("Toilets", "Toilets", "amenities", 7.5, -1.6, door=True),
        P("Medical Imaging", "Medical Imaging", "imaging", 10.4, 0.0, door=True),
        P("Main Reception", "Main Reception", "amenities", -10.4, 0.0, area=True),
    ]


def _office() -> list[Place]:
    """The office (Isaac Sim's office floor plan): meeting rooms down the middle, open-plan desks either
    side, the reception and offices to the west, the lunch room and two more meeting rooms to the south."""
    P = Place
    return [
        # North Wing
        P("Kitchen", "Kitchen", "amenities", -4.0, 44.5, "E"),
        P("Quiet Zone", "Quiet Zone", "office", -14.2, 44.0, "W", area=True),
        P("Breakout Area", "Breakout Area", "amenities", 14.0, 40.5, "E", area=True),
        P("Boardroom", "Boardroom", "office", -4.0, 34.0, "E", door=True),
        P("Training Room", "Training Room", "office", -4.0, 25.0, "E", door=True),
        P("Toilets", "Toilets", "amenities", 5.8, 31.5, "W", door=True),
        P("Accessible Toilet", "Toilets", "amenities", 5.8, 36.8, "W", door=True),
        P("Open Plan North", "Open Plan North", "office", -14.2, 27.5, "W", area=True),
        P("Hot Desks", "Hot Desks", "office", 14.0, 26.0, "E", area=True),
        # Central
        P("Main Reception", "Main Reception", "amenities", -14.2, 12.8, "W", area=True),
        P("Records Room", "Records Room", "staff", 2.5, 12.2, "S", staff=True, door=True),
        P("Conference Room", "Conference Room", "office", 2.5, -2.8, "N", door=True),
        P("Manager's Office", "Manager's Office", "staff", -8.0, 8.5, "W", staff=True, door=True),
        P("HR Office", "HR Office", "staff", -8.0, 4.0, "W", staff=True, door=True),
        P("Print Room", "Print Room", "office", -7.0, -0.5, "W"),
        P("Comms Room", "Comms Room", "staff", -10.0, -0.5, "W", staff=True),
        P("Plant Room", "Plant Room", "staff", -10.5, -5.6, "S", staff=True, door=True),
        P("Open Plan Central", "Open Plan Central", "office", 14.0, 5.0, "E", area=True),
        P("Lifts", "Lifts", "amenities", 5.4, 8.0, "W", door=True),
        P("Stairs", "Stairs", "amenities", 5.4, 1.5, "W", door=True),
        # South Wing
        P("Lunch Room", "Lunch Room", "amenities", -10.0, -11.5, "S", door=True),
        P("Open Plan South", "Open Plan South", "office", 13.0, -23.5, "E", area=True),
        P("Cleaner's Store", "Cleaner's Store", "staff", 12.0, -20.0, "E", staff=True),
        P("Meeting Room 1", "Meeting Room 1", "office", 4.5, -24.2, "S", door=True),
        P("Meeting Room 2", "Meeting Room 2", "office", -1.5, -34.5, "E", door=True),
        P("Fire Stairs", "Fire Stairs", "staff", -1.5, -39.2, "S", staff=True, door=True),
    ]


def _house() -> list[Place]:
    """The house: a sign in each room, as dementia-friendly homes have (no directory: a home has none).
    Rooms from what is in them: the bed and wardrobe, the island bench and corner counters, the table
    with eight chairs."""
    P = Place
    return [
        P("Bedroom", "Bedroom", "home", -8.8, -0.3, "W", area=True),
        P("Study", "Study", "home", -8.8, -3.0, "W", area=True),
        P("Living Room", "Living Room", "home", 1.0, 4.7, "N", area=True),
        P("Dining Room", "Dining Room", "home", 8.6, 2.0, "E", area=True),
        P("Kitchen", "Kitchen", "home", 0.5, -4.4, "S", area=True),
    ]


HAND_NAMED = {"fp_hospital": _hospital, "corridor": _corridor, "fp_office": _office, "fp_house": _house}
WINGS = {                 # which wing each department (or place) is in
    "fp_hospital": {
        "West Wing": ["Ward 4A", "Emergency"],
        "Central Wing": ["Main Reception", "Cafe", "Medical Imaging", "Pharmacy", "Pathology", "Outpatient Clinics",
                    "Waiting Area"],
        "East Wing": ["Ward 4B", "Allied Health", "Toilets", "Lifts"],
    },
    "fp_office": {
        "North Wing": ["Kitchen", "Quiet Zone", "Breakout Area", "Boardroom", "Training Room", "Toilets",
                       "Open Plan North", "Hot Desks"],
        "Central Wing": ["Main Reception", "Records Room", "Conference Room", "Manager's Office", "HR Office",
                    "Print Room", "Comms Room", "Plant Room", "Open Plan Central", "Lifts", "Stairs"],
        "South Wing": ["Lunch Room", "Open Plan South", "Cleaner's Store", "Meeting Room 1", "Meeting Room 2",
                       "Fire Stairs"],
    },
}
NO_DIRECTORY = {"fp_house"}   # homes have none
ROOM_NAMES = {            # for buildings named by size: the biggest room first
    "office": ["Open Plan Office", "Meeting Room 1", "Meeting Room 2", "Meeting Room 3", "Kitchen", "Print Room",
               "Toilets", "Storeroom", "Quiet Room", "Comms Room", "Meeting Room 4", "Meeting Room 5"],
    "warehouse": ["Office", "Amenities", "First Aid", "Storeroom"],
}
WALL_NAMES = {            # for buildings named along their outer walls: evenly spread
    "office": ["Main Reception", "Lifts", "Stairs", "Kitchen", "Toilets"],
    "warehouse": ["Loading Dock 1", "Loading Dock 2", "Loading Dock 3", "Loading Dock 4", "Goods In", "Dispatch",
                  "Forklift Charging", "Office", "First Aid", "Amenities"],
}
GROUP_OF = {"Loading Dock": "Loading Docks", "Meeting Room": "Meeting Rooms"}


def building_type(building: str) -> str | None:
    if building in HAND_NAMED:
        return "hand"
    if "office" in building:
        return "office"
    if "warehouse" in building or "depot" in building:
        return "warehouse"
    return None


def wing_of(building: str) -> dict:
    """Department (or place) -> its wing, for a building with wings; empty for one without."""
    return {g: wing for wing, groups in WINGS.get(building, {}).items() for g in groups}


def group_of(name: str) -> str:
    for prefix, group in GROUP_OF.items():
        if name.startswith(prefix):
            return group
    return name


# ---------------------------------------------------------------------------------- the floor plan
class Geometry:
    """Rooms, doors, walls and junctions of a floor plan, in the plan's pixels and metres."""

    def __init__(self, plan):
        self.plan, self.res = plan, plan.resolution
        self.free = plan.cells == FREE
        self.wall = plan.cells == WALL
        self._regions()
        self._closed_rooms()

    # coordinates
    def px(self, x, y):
        return self.plan._pixel(x, y)

    def xy(self, row, col):
        return (col + 0.5) * self.res - self.plan.width_m / 2, self.plan.height_m / 2 - (row + 0.5) * self.res

    def is_wall(self, x, y) -> bool:
        r, c = self.px(x, y)
        return bool(self.wall[r, c]) or abs(x) >= self.plan.width_m / 2 or abs(y) >= self.plan.height_m / 2

    def is_free(self, x, y) -> bool:
        r, c = self.px(x, y)
        return bool(self.free[r, c])

    def _regions(self):
        """Rooms and corridors: open floor split at doorways (narrower than about 1.1 m)."""
        free8 = self.free.astype(np.uint8)
        r = int(round(0.55 / self.res))
        core = cv2.erode(free8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        markers = np.zeros(self.free.shape, np.int32)
        keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] * self.res ** 2 > 1.0]
        for i, k in enumerate(keep, start=1):
            markers[labels == k] = i
        outside = len(keep) + 1
        markers[~self.free] = outside
        cv2.watershed(cv2.cvtColor(free8 * 255, cv2.COLOR_GRAY2BGR), markers)
        markers[(markers == outside) | (markers < 0) | ~self.free] = 0
        self.labels = markers
        areas = np.bincount(markers.ravel(), minlength=outside)
        areas[0] = 0
        self.areas = areas * self.res ** 2
        self.main = int(np.argmax(self.areas))           # the corridors and halls: the biggest open region

    def _closed_rooms(self):
        """Rooms with no door on the plan: unknown areas inside the building, walled all round."""
        unknown = (self.plan.cells == UNKNOWN).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(unknown, connectivity=4)
        edge = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
        self.closed_labels = labels
        self.closed = {k: stats[k] for k in range(1, n)
                       if k not in edge and stats[k, cv2.CC_STAT_AREA] * self.res ** 2 >= 4.0}

    # walls
    def march(self, x, y, dx, dy, max_m, until_wall=True):
        """Walk from (x, y) along (dx, dy) until a wall pixel (or until leaving the walls); returns the
        distance walked, or None if nothing within max_m."""
        step = self.res / 2
        d = 0.0
        while d <= max_m:
            if self.is_wall(x + dx * d, y + dy * d) == until_wall:
                return d
            d += step
        return None

    def wall_face(self, x, y, nx, ny, max_m=1.5):
        """The point on the wall face in front of (x, y) when looking along -(nx, ny), moved WALL_OFFSET_M
        out of the wall, or None."""
        d = self.march(x, y, -nx, -ny, max_m)
        if d is None:
            return None
        return x - nx * d + nx * WALL_OFFSET_M, y - ny * d + ny * WALL_OFFSET_M

    def nearest_wall(self, x, y, max_m=4.0):
        """(distance, normal pointing away from the wall) to the nearest wall from (x, y)."""
        best = None
        for a in np.linspace(0, 2 * math.pi, 72, endpoint=False):
            d = self.march(x, y, math.cos(a), math.sin(a), max_m)
            if d is not None and (best is None or d < best[0]):
                best = (d, -math.cos(a), -math.sin(a))
        if best is None:
            return None
        d, nx, ny = best
        # square the normal up with the wall: the direction of most wall pixels around the hit point
        return d, nx, ny

    def door_of(self, region: int):
        """The doorway between a room and the region it opens onto: (centre x, y, normal into that
        region, half width), or None."""
        mask = (self.labels == region).astype(np.uint8)
        grown = cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)
        touch = grown & (self.labels > 0) & (self.labels != region)
        if not touch.any():
            return None
        n, lab, stats, cents = cv2.connectedComponentsWithStats(touch.astype(np.uint8), connectivity=8)
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        rows, cols = np.nonzero(lab == k)
        pts = np.stack([(cols + 0.5) * self.res - self.plan.width_m / 2,
                        self.plan.height_m / 2 - (rows + 0.5) * self.res], axis=1)
        centre = pts.mean(axis=0)
        ry, rx = np.nonzero(mask)
        room_centre = np.array([((rx + 0.5) * self.res - self.plan.width_m / 2).mean(),
                                (self.plan.height_m / 2 - (ry + 0.5) * self.res).mean()])
        if len(pts) >= 3:
            _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
            along = vecs[:, -1]
            normal = np.array([-along[1], along[0]])
        else:
            normal = centre - room_centre
        normal = normal / (np.linalg.norm(normal) or 1.0)
        if normal @ (centre - room_centre) < 0:
            normal = -normal
        nx, ny = _square_up(*normal)
        # the opening's edges: walk sideways from just inside the doorway
        half = []
        for s in (1, -1):
            d = self.march(centre[0] - nx * 0.05, centre[1] - ny * 0.05, -ny * s, nx * s, 2.0)
            half.append(d if d is not None else 0.5)
        return float(centre[0]), float(centre[1]), nx, ny, half

    def closed_side(self, k: int):
        """The side of a closed room facing the most open floor: (door centre x, y, normal out)."""
        x0, y0, w, h, _ = self.closed[k]
        best = None
        for side, (nx, ny) in enumerate([(0, 1), (0, -1), (-1, 0), (1, 0)]):    # top, bottom, left, right
            if ny:
                row = y0 - 1 if ny > 0 else y0 + h
                line = [(row - ny * int(0.6 / self.res), c) for c in range(x0, x0 + w)]
            else:
                col = x0 - 1 if nx < 0 else x0 + w
                line = [(r, col + nx * int(0.6 / self.res)) for r in range(y0, y0 + h)]
            ok = [(r, c) for r, c in line if 0 <= r < self.free.shape[0] and 0 <= c < self.free.shape[1]
                  and self.free[r, c] and self.plan.clearance[r, c] >= 0.45]
            if best is None or len(ok) > len(best[0]):
                best = (ok, nx, ny)
        ok, nx, ny = best
        if len(ok) < int(0.8 / self.res):
            return None
        r, c = ok[len(ok) // 2]
        x, y = self.xy(r, c)
        return x, y, float(nx), float(ny)


def _square_up(nx, ny, within_deg=20.0):
    """Snap a direction to the nearest axis if it is close to one (walls are mostly square)."""
    a = math.atan2(ny, nx)
    snapped = round(a / (math.pi / 2)) * (math.pi / 2)
    if abs(math.atan2(math.sin(a - snapped), math.cos(a - snapped))) < math.radians(within_deg):
        a = snapped
    return math.cos(a), math.sin(a)


# ---------------------------------------------------------------------------------- layout
@dataclass
class Layout:
    building: str
    faces: list
    doors: list                    # (x, y, yaw) of doors drawn on walls
    boards: list                   # (x, y, z, yaw, w, h) backing boards
    stops: dict                    # place name -> (x, y) the chair stops at

    def by_tag(self) -> dict:
        return {f.tag: f for f in self.faces}

    def places(self) -> list[str]:
        return sorted({f.place for f in self.faces if f.place})


def _mount(geo: Geometry, place: Place, fx, fy, nx, ny, half=(0.5, 0.5), draw_door=False, centred=False):
    """Signs for a door (or a wall spot) at (fx, fy) whose corridor side looks along (nx, ny):
    the plate on the handle side, the blade sign above it, and a door drawn on the wall if asked.
    Returns (plate pose, blade pose, door pose, stop) or None."""
    px_, py_ = -ny, nx                       # along the wall
    out = []
    half_plate = KINDS["plate"]["w"] / 2
    for side in (1, -1):                     # handle side first, the other if there is no wall there
        offset = (half[0] if side == 1 else half[1]) if not draw_door else DOOR_W / 2
        for extra in (0.0, 0.3, 0.6):        # slide along the wall past anything in front (an open door leaf)
            along = (extra * 1.3) if centred else offset + 0.10 + half_plate + extra
            sx, sy = fx + px_ * side * along + nx * 0.6, fy + py_ * side * along + ny * 0.6
            face = geo.wall_face(sx, sy, nx, ny, max_m=1.4)
            if face is None:
                continue
            # the whole plate must have wall behind it and open floor in front of it
            ends = [geo.march(face[0] + px_ * t + nx * 0.2, face[1] + py_ * t + ny * 0.2, -nx, -ny, 0.6)
                    for t in (-half_plate, half_plate)]
            front = [geo.plan.clearance_at(face[0] + px_ * t + nx * d, face[1] + py_ * t + ny * d)
                     for t in (-half_plate, 0.0, half_plate) for d in (0.25, 0.5, 0.8)]
            if any(e is None for e in ends) or min(front) < 0.1:
                continue
            out = [face, side]
            break
        if out:
            break
    if not out:
        return None
    (wx, wy), side = out
    yaw = math.atan2(ny, nx)
    blade_reach = WALL_OFFSET_M + KINDS["blade"]["w"] / 2
    blade = (wx + nx * blade_reach - px_ * side * 0.15, wy + ny * blade_reach - py_ * side * 0.15)
    door = None
    if draw_door:
        dface = geo.wall_face(fx + nx * 0.6, fy + ny * 0.6, nx, ny, max_m=1.4)
        if dface is not None:
            door = (dface[0] - nx * 0.035, dface[1] - ny * 0.035, yaw)
    stop = _stop_near(geo, fx, fy, wx, wy, nx, ny)
    return (wx, wy, yaw), blade, door, stop


def _stop_near(geo: Geometry, fx, fy, wx, wy, nx, ny):
    """Where the chair stops for a place: a metre out from its door, or the nearest spot with room
    for the chair close to it (a cafe's tables, a trolley parked by the door)."""
    best = None
    for out in (1.0, 1.3, 1.6, 2.0, 2.5):
        for t in (0.0, 0.3, -0.3, 0.6, -0.6, 1.0, -1.0, 1.5, -1.5):
            for bx, by in ((fx, fy), (wx, wy)):
                x, y = bx + nx * out - ny * t, by + ny * out + nx * t
                if geo.plan.clearance_at(x, y) >= 0.55:
                    cost = out + abs(t)
                    if best is None or cost < best[0]:
                        best = (cost, (x, y))
        if best is not None:
            return best[1]
    return fx + nx * 1.0, fy + ny * 1.0


def _auto_places(geo: Geometry, kind: str) -> list[Place]:
    """Names for a building without a hand-made list: its rooms by size, then spots along its walls."""
    colour = "office" if kind == "office" else "warehouse"
    places, used = [], set()
    rooms = [k for k in range(1, len(geo.areas)) if k != geo.main and 6.0 <= geo.areas[k] <= 400.0
             and geo.door_of(k) is not None]
    rooms += [-k for k in geo.closed if geo.closed_side(k) is not None and geo.closed[k][4] * geo.res ** 2 <= 200]
    def size(k):
        return geo.areas[k] if k > 0 else geo.closed[-k][4] * geo.res ** 2
    names = iter(ROOM_NAMES[kind])
    for k in sorted(rooms, key=size, reverse=True):
        name = next(names, None)
        if name is None:
            break
        if k > 0:
            rows, cols = np.nonzero(geo.labels == k)
        else:
            rows, cols = np.nonzero(geo.closed_labels == -k)
        i = len(rows) // 2
        x, y = geo.xy(int(np.median(rows)), int(np.median(cols)))
        if (k > 0 and geo.labels[geo.px(x, y)] != k) or (k < 0 and geo.closed_labels[geo.px(x, y)] != -k):
            x, y = geo.xy(int(rows[i]), int(cols[i]))
        places.append(Place(name, group_of(name), colour, x, y))
        used.add(name)
    if kind == "warehouse":                       # aisle signs on both ends of each long run of shelves
        places += _aisles(geo)
    # spots along the main region's outer walls, evenly spread
    main = (geo.labels == geo.main).astype(np.uint8)
    contours, _ = cv2.findContours(main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=len)[:, 0, :]
        wall_names = [n for n in WALL_NAMES[kind] if n not in used]
        step = max(1, len(contour) // max(1, len(wall_names)))
        for i, name in enumerate(wall_names):
            c, r = contour[(i * step + step // 2) % len(contour)]
            x, y = geo.xy(int(r), int(c))
            hit = geo.nearest_wall(x, y, 1.0)
            if hit is None:
                continue
            d, nx, ny = hit
            x, y = x + nx * 0.8, y + ny * 0.8
            if geo.plan.clearance_at(x, y) < 0.5:
                continue
            places.append(Place(name, group_of(name), "safety" if name in ("First Aid", "Amenities", "Toilets")
                                else colour, x, y, door=True))
    return places


WALL_SIDES = {"N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0)}


def _opening(geo: Geometry, x, y, wx, wy, depth, reach=2.0):
    """A doorway in the wall on side (wx, wy) of the spot (x, y), whose face is `depth` away: the
    opening nearest the spot, at least 0.6 m wide, as (offset along the wall, half widths), or None."""
    ax, ay = -wy, wx                                      # along the wall
    steps = np.arange(-reach, reach + 1e-9, geo.res)
    open_ = [geo.march(x + ax * t, y + ay * t, wx, wy, depth + 0.35) is None for t in steps]
    runs, start = [], None
    for i, o in enumerate(open_ + [False]):
        if o and start is None:
            start = i
        elif not o and start is not None:
            runs.append((steps[start], steps[i - 1]))
            start = None
    runs = [(a, b) for a, b in runs if b - a >= 0.6 and a > -reach and b < reach]
    if not runs:
        return None
    a, b = min(runs, key=lambda ab: abs((ab[0] + ab[1]) / 2))
    mid = (a + b) / 2
    return mid, ((b - a) / 2, (b - a) / 2)


def _at_wall(geo: Geometry, place: Place):
    """Signs for a place given as a corridor spot and a wall side: at its doorway if the wall has
    one near the spot, else on a door drawn on the wall (a closed room), or centred (an area)."""
    wx, wy = WALL_SIDES[place.wall]
    nx, ny = -wx, -wy
    # how far the wall is: the nearest hit in a band either side of the spot (straight ahead may be the doorway)
    hits = [geo.march(place.x - wy * t, place.y + wx * t, wx, wy, 3.0) for t in np.arange(-2.0, 2.01, 0.25)]
    hits = [h for h in hits if h is not None]
    if not hits:
        return None
    depth = float(np.percentile(hits, 25))
    fx, fy = place.x + wx * depth, place.y + wy * depth
    if place.area:
        return _mount(geo, place, fx, fy, nx, ny, centred=True)
    gap = None if place.door else _opening(geo, place.x, place.y, wx, wy, depth)
    if gap is None:
        return _mount(geo, place, fx, fy, nx, ny, draw_door=True)
    mid, half = gap
    ax, ay = -wy, wx
    return _mount(geo, place, fx + ax * mid, fy + ay * mid, nx, ny, half)


def _near(geo: Geometry, place: Place):
    """Signs for a place given only as a spot: on the nearest wall (a door drawn there if asked)."""
    hit = geo.nearest_wall(place.x, place.y, 6.0)
    if hit is None:
        return None
    d, nx, ny = hit
    nx, ny = _square_up(nx, ny)
    return _mount(geo, place, place.x - nx * d, place.y - ny * d, nx, ny, draw_door=place.door,
                  centred=place.area)


def _aisles(geo: Geometry) -> list[Place]:
    """Long, narrow blocks standing free in the building (rows of shelving): "Aisle A", "Aisle B"...
    on both ends, in order across the building."""
    walls = geo.wall.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(walls, connectivity=8)
    edge = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    runs = []
    for k in range(1, n):
        x0, y0, w, h, area = stats[k]
        long_, short = max(w, h) * geo.res, min(w, h) * geo.res
        if k in edge or long_ < 4.0 or long_ < 3 * short or area < 0.5 * w * h:
            continue
        if w >= h:                                 # along x: ends to the west and east
            row = y0 + h // 2
            ends = [(geo.xy(row, x0)[0] - 0.8, geo.xy(row, x0)[1], "E"), (geo.xy(row, x0 + w - 1)[0] + 0.8, geo.xy(row, x0)[1], "W")]
        else:
            col = x0 + w // 2
            ends = [(geo.xy(y0, col)[0], geo.xy(y0, col)[1] + 0.8, "S"), (geo.xy(y0 + h - 1, col)[0], geo.xy(y0 + h - 1, col)[1] - 0.8, "N")]
        runs.append((geo.xy(y0 + h // 2, x0 + w // 2), ends))
    places = []
    for i, (_, ends) in enumerate(sorted(runs, key=lambda r: (round(r[0][1] / 3), r[0][0]))):
        if i >= 26:
            break
        for x, y, wall in ends:
            if geo.plan.clearance_at(x, y) >= 0.5:
                places.append(Place(f"Aisle {chr(65 + i)}", "Aisles", "warehouse", x, y, wall, area=True))
    return places


def make_layout(plan, building: str) -> Layout:
    kind = building_type(building)
    geo = Geometry(plan)
    if kind is None:
        return Layout(building, [], [], [], {})
    if kind == "hand":
        places = HAND_NAMED[building]()
    else:                                           # named on the building itself, so its variants match
        base = json.loads((WEBOTS_DIR / "worlds" / f"{building}.json").read_text())["map"]
        base_plan = load_floorplan(str((ROOT / base).resolve()))
        places = _auto_places(geo if base_plan.cells.shape == plan.cells.shape and (base_plan.cells == plan.cells).all()
                              else Geometry(base_plan), kind)
    wings = wing_of(building)
    plates, blades, doors, boards, stops = [], [], [], [], {}

    def free(mount, gap) -> bool:
        """Not on top of another door's signs (a plate is 0.5 m wide, a blade 0.72 m)."""
        if mount is None:
            return False
        (wx, wy, _), (bx, by), _, _ = mount
        return all(math.dist((wx, wy), (f.x, f.y)) >= gap for f in plates) and \
            all(math.dist((bx, by), (f.x, f.y)) >= gap for f in blades)

    def mount_for(place, gap):
        """Its own spot, or (furniture or another door's signs in the way) a little along the wall."""
        if place.wall:
            wx, wy = WALL_SIDES[place.wall]
            ways = [(-wy, wx)]
        else:
            ways = [(1.0, 0.0), (0.0, 1.0)]
        for shift in (0.0, 0.6, -0.6, 1.2, -1.2, 1.8, -1.8, 2.4, -2.4):
            for ax, ay in ways:
                moved = Place(**{**asdict(place), "x": place.x + ax * shift, "y": place.y + ay * shift})
                mount = _at_wall(geo, moved) if place.wall else _near(geo, moved)
                if free(mount, gap):
                    return mount
        return None

    def hang(place) -> bool:
        mount = mount_for(place, SIGN_GAP_M) or mount_for(place, MIN_SIGN_GAP_M)   # tight spots: a little closer
        if mount is None:
            return False
        (wx, wy, yaw), (bx, by), door, stop = mount
        stops.setdefault(place.name, (round(stop[0], 3), round(stop[1], 3)))
        common = dict(place=place.name, group=place.group, colour=place.colour, staff=place.staff,
                      stop=(round(stop[0], 3), round(stop[1], 3)))
        plates.append(Face(0, "plate", wx, wy, KINDS["plate"]["z"], yaw, **common))
        along = yaw + math.pi / 2                          # the blade's faces look along the wall
        for s_ in (0, 1):
            off = 0.006 * (1 if s_ == 0 else -1)
            blades.append(Face(0, "blade", bx + math.cos(along) * off, by + math.sin(along) * off,
                               KINDS["blade"]["z"], along + math.pi * s_, **common))
        boards.append((bx, by, KINDS["blade"]["z"], along, KINDS["blade"]["w"], KINDS["blade"]["h"]))
        boards.append((wx - math.cos(yaw) * 0.006, wy - math.sin(yaw) * 0.006, KINDS["plate"]["z"], yaw,
                       KINDS["plate"]["w"], KINDS["plate"]["h"]))
        if door is not None:
            doors.append(door)
        return True

    for place in places:
        if not hang(place):
            print(f"signs: no wall for {place.name} in {building}", file=sys.stderr)
    if wings:                                       # a department with several rooms: its own sign at its entrance
        for entrance in _entrances(geo, places, stops):
            if hang(entrance):
                places.append(entrance)
    directories = [] if building in NO_DIRECTORY else _directory_faces(geo, places, stops, wings, plates)
    directions = _direction_faces(geo, places, stops, wings)
    faces = []
    for kind_name, items in (("plate", plates), ("blade", blades), ("direction", directions),
                             ("directory", directories)):
        first, count = KINDS[kind_name]["first"], KINDS[kind_name]["count"]
        if len(items) > count:
            print(f"signs: {len(items)} {kind_name} faces in {building}, only {count} tags; dropping the rest",
                  file=sys.stderr)
            items = items[:count]
        for i, f in enumerate(items):
            f.tag = first + i
            faces.append(f)
            if kind_name in ("direction", "directory"):
                boards.append((f.x - math.cos(f.yaw) * 0.006, f.y - math.sin(f.yaw) * 0.006, f.z, f.yaw,
                               KINDS[kind_name]["w"], KINDS[kind_name]["h"]))
    return Layout(building, faces, doors, boards, stops)


def entries(places: list[Place]) -> list[tuple[str, str]]:
    """What a directory lists: each department with a sign of its own name (a ward's entrance sign,
    or a place called the same), else its places one by one; staff rooms left out. Emergency first,
    then A to Z, as hospital directories do. [(name, group)]."""
    named = {p.name for p in places}
    out, seen = [], set()
    for p in places:
        if p.staff:
            continue
        name = p.group if p.group in named else p.name
        if name not in seen:
            seen.add(name)
            out.append((name, p.group))
    return sorted(out, key=lambda e: (e[0] != "Emergency", e[0].lower()))


def _entrances(geo: Geometry, places: list[Place], stops: dict) -> list[Place]:
    """A sign for each department with several rooms and no place of its name: on the wall beside its
    room nearest the main reception, a little towards the reception (seen first on the way in)."""
    named = {p.name for p in places}
    reception = stops.get("Main Reception")
    groups: dict[str, list[Place]] = {}
    for p in places:
        if not p.staff and p.name in stops and p.wall:
            groups.setdefault(p.group, []).append(p)
    out = []
    for group, members in groups.items():
        if len(members) < 2 or group in named:
            continue
        field_ = _distance_field(geo.plan, [reception]) if reception else None
        def dist(p):
            cell = geo.plan._nearest_open(geo.plan.cell_of(*stops[p.name]))
            return field_[cell] if field_ is not None and cell is not None else 0.0
        first = min(members, key=dist)
        wx, wy = WALL_SIDES[first.wall]
        ax, ay = -wy, wx                                   # along the wall
        toward = 1.0 if reception is None or (reception[0] - first.x) * ax + (reception[1] - first.y) * ay >= 0 else -1.0
        out.append(Place(group, group, first.colour, first.x + ax * toward * 1.9, first.y + ay * toward * 1.9,
                         first.wall, area=True, entrance=True))
    return out


def _mount_board(geo: Geometry, x, y, avoid, kind="directory"):
    """A spot on a wall near (x, y) for a board: the wall behind all of it, open floor in front with no
    wall sticking out beside it, room for the chair to stop and read it, and no other sign close by.
    (face x, y, yaw, stop) or None."""
    k = KINDS[kind]
    half = k["w"] / 2 + 0.05
    best = None
    for a in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        d = geo.march(x, y, math.cos(a), math.sin(a), 5.0)
        if d is None:
            continue
        nx, ny = _square_up(-math.cos(a), -math.sin(a))
        hx, hy = x + math.cos(a) * d, y + math.sin(a) * d
        fx, fy = hx + nx * WALL_OFFSET_M, hy + ny * WALL_OFFSET_M
        ax, ay = -ny, nx
        ok = True
        for t in np.linspace(-half, half, 9):
            back = geo.march(fx + ax * t + nx * 0.2, fy + ay * t + ny * 0.2, -nx, -ny, 0.6)
            if back is None or any(geo.plan.clearance_at(fx + ax * t + nx * o, fy + ay * t + ny * o) < 0.12
                                   for o in (0.4, 0.9, 1.5)):
                ok = False
                break
        middle = geo.march(fx + nx * 0.3, fy + ny * 0.3, -nx, -ny, 1.0)
        for t in np.arange(-half - 0.25, half + 0.25, geo.res):   # no wall sticking out beside it or just past
            if not ok or middle is None:                    # its ends (3D walls are 10 cm blocks, so even a
                ok = False                                  # few cm on the plan can hide its edge)
                break
            depth = geo.march(fx + ax * t + nx * 0.3, fy + ay * t + ny * 0.3, -nx, -ny, 1.0)
            ok = depth is None or depth > middle - 0.04
        stop = (fx + nx * 1.8, fy + ny * 1.8)
        if not ok or geo.plan.clearance_at(*stop) < 0.5:
            continue
        if any(math.dist((fx, fy), q) < half + 0.6 for q in avoid):
            continue
        if best is None or d < best[0]:
            best = (d, fx, fy, math.atan2(ny, nx), stop)
    return best[1:] if best else None


def _directory_faces(geo: Geometry, places: list[Place], stops: dict, wings: dict, plates) -> list[Face]:
    """Directories: one at the main reception, then at the junction furthest from any directory, until
    every junction is within DIRECTORY_EVERY_M of one. Each lists the building's entries and wings."""
    plan = geo.plan
    listed = entries([p for p in places if p.name in stops])
    if not listed:
        return []
    rows = [(name, wings.get(group, "")) for name, group in listed][:2 * DIRECTORY_ROWS]
    groups: dict[str, list] = {}
    for p in places:
        if p.name in stops:
            groups.setdefault(p.group, []).append(stops[p.name])
    fields = {g: _distance_field(plan, pts) for g, pts in groups.items()} if wings else {}
    reception = stops.get("Main Reception") or stops.get("Goods In") or next(iter(stops.values()))
    spots = [reception]
    junctions = [(x, y) for x, y, _ in _junctions(plan, geo)]
    field_ = _distance_field(plan, spots)
    while len(spots) < MAX_DIRECTORIES and junctions:
        def far(j):
            cell = plan._nearest_open(plan.cell_of(*j))
            return field_[cell] if cell is not None and np.isfinite(field_[cell]) else -1.0
        j = max(junctions, key=far)
        if far(j) <= DIRECTORY_EVERY_M:
            break
        spots.append(j)
        junctions.remove(j)
        field_ = np.minimum(field_, _distance_field(plan, [j]))
    avoid = [(f.x, f.y) for f in plates]
    faces = []
    for sx, sy in spots:
        mount = None
        for r in (0.0, 2.5, 4.0, 6.0):              # on the nearest free wall: other signs may hang right there
            for a in (np.linspace(0, 2 * math.pi, 8, endpoint=False) if r else [0.0]):
                x, y = sx + r * math.cos(a), sy + r * math.sin(a)
                if plan.clearance_at(x, y) >= 0.5:
                    mount = _mount_board(geo, x, y, avoid)
                if mount is not None:
                    break
            if mount is not None:
                break
        if mount is None:
            continue
        fx, fy, yaw, stop = mount
        avoid.append((fx, fy))
        here = ""
        if fields:
            cell = plan._nearest_open(plan.cell_of(*stop))
            if cell is not None:
                nearest = min(fields, key=lambda g: fields[g][cell])
                here = wings.get(nearest, "")
        faces.append(Face(0, "directory", fx, fy, KINDS["directory"]["z"], yaw, rows=list(rows), here=here,
                          stop=(round(stop[0], 3), round(stop[1], 3))))
    return faces


# ---------------------------------------------------------------------------------- direction signs
def _skeleton(open_grid: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning of a boolean grid."""
    img = open_grid.astype(np.uint8).copy()
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 0
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2, P3, P4, P5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            P6, P7, P8, P9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            ring = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(ring[:-1])
            A = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c1, c2 = P2 * P4 * P6, P4 * P6 * P8
            else:
                c1, c2 = P2 * P4 * P8, P2 * P6 * P8
            kill = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            return img.astype(bool)


def _distance_field(plan, sources) -> np.ndarray:
    """Route-grid distance (m) from every open cell to the nearest of `sources` (x, y)."""
    blocked = plan.blocked
    dist = np.full(blocked.shape, np.inf)
    heap = []
    for x, y in sources:
        cell = plan._nearest_open(plan.cell_of(x, y))
        if cell is not None:
            dist[cell] = 0.0
            heap.append((0.0, cell))
    heapq.heapify(heap)
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    rows, cols = blocked.shape
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr, dc, w in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not blocked[nr, nc]:
                nd = d + w * ROUTE_CELL_M
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(heap, (nd, (nr, nc)))
    return dist


def _downhill(plan, field_, start, metres=3.0):
    """Follow a distance field downhill from cell `start` for `metres`: the way to go, as a unit
    (dx, dy), or None."""
    r, c = start
    if not np.isfinite(field_[r, c]):
        return None
    x0, y0 = plan.centre_of(r, c)
    walked = 0.0
    rows, cols = field_.shape
    while walked < metres and field_[r, c] > 0:
        best = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (dr or dc) and 0 <= nr < rows and 0 <= nc < cols and field_[nr, nc] < field_[r, c]:
                    if best is None or field_[nr, nc] < field_[best]:
                        best = (nr, nc)
        if best is None:
            break
        walked += ROUTE_CELL_M * (1.4142 if best[0] != r and best[1] != c else 1.0)
        r, c = best
    x1, y1 = plan.centre_of(r, c)
    n = math.hypot(x1 - x0, y1 - y0)
    return ((x1 - x0) / n, (y1 - y0) / n) if n > 0.3 else None


def _junctions(plan, geo: Geometry):
    """Corridor junctions: skeleton points where at least three corridors at least 1.4 m wide meet.
    Returns [(x, y, [branch directions])]."""
    open_ = ~plan.blocked
    skel = _skeleton(open_)
    rows, cols = skel.shape
    nb = sum(np.roll(np.roll(skel, dr, 0), dc, 1).astype(int) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if dr or dc)
    points = np.argwhere(skel & (nb >= 3))
    found = []
    for r, c in points:
        x, y = plan.centre_of(r, c)
        branches = []
        for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            dx, dy = math.cos(a), math.sin(a)
            ok = True
            for t in np.arange(1.0, 3.01, 0.5):     # a corridor (1.5 to 5 m wide) for 3 m, starting 1 m out
                if not 0.75 <= plan.clearance_at(x + dx * t, y + dy * t) <= 2.5:
                    ok = False
                    break
            if ok:
                branches.append(a)
        # group nearby angles into branches
        groups = []
        for a in branches:
            if groups and abs(math.atan2(math.sin(a - groups[-1][-1]), math.cos(a - groups[-1][-1]))) <= math.radians(23):
                groups[-1].append(a)
            else:
                groups.append([a])
        if len(groups) > 1 and abs(math.atan2(math.sin(groups[0][0] - groups[-1][-1]),
                                              math.cos(groups[0][0] - groups[-1][-1]))) <= math.radians(23):
            groups[0] = groups[-1] + groups[0]
            groups.pop()
        if not 3 <= len(groups) <= 4 or any(len(g) > 6 for g in groups) or plan.clearance_at(x, y) > 2.5:
            continue                                 # a hall, not a junction
        dirs = [math.atan2(sum(math.sin(a) for a in g), sum(math.cos(a) for a in g)) for g in groups]
        found.append((x, y, dirs))
    merged = []
    for x, y, dirs in found:                         # one junction per 8 m
        if all(math.dist((x, y), (mx, my)) > 8.0 for mx, my, _ in merged):
            merged.append((x, y, dirs))
    return merged


def _direction_faces(geo: Geometry, places: list[Place], stops: dict, wings: dict | None = None) -> list[Face]:
    """A face in each corridor at each junction, with up to MAX_ROWS destinations and their arrows. With
    wings: the other wings first (at most two), then the departments of the wing the junction is in."""
    plan = geo.plan
    wings = wings or {}
    groups: dict[str, list] = {}
    for p in places:
        if not p.staff and p.name in stops:
            groups.setdefault(p.group, []).append(stops[p.name])
    if len(groups) < 2:
        return []
    fields = {g: _distance_field(plan, pts) for g, pts in groups.items()}
    wing_fields = {}
    for wing in sorted(set(wings.values())):
        pts = [pt for g, pt_list in groups.items() if wings.get(g) == wing for pt in pt_list]
        if pts:
            wing_fields[wing] = _distance_field(plan, pts)
    faces = []
    for x, y, dirs in _junctions(plan, geo):
        cell = plan._nearest_open(plan.cell_of(x, y))
        if cell is None:
            continue
        for d in dirs:                               # a face in each corridor, looking down it
            half = KINDS["direction"]["w"] / 2 + 0.1
            spot = None
            for out_m in (1.6, 2.2, 2.8):            # the whole sign must hang between the walls
                fx, fy = x + math.cos(d) * out_m, y + math.sin(d) * out_m
                for shift in (0.0, 0.3, -0.3, 0.6, -0.6):
                    cx, cy = fx - math.sin(d) * shift, fy + math.cos(d) * shift
                    ends = [(cx - math.sin(d) * e, cy + math.cos(d) * e) for e in (-half, half)]
                    if plan.clearance_at(cx, cy) >= 0.6 and all(plan.clearance_at(*q) >= 0.1 for q in ends):
                        spot = (cx, cy)
                        break
                if spot:
                    break
            if spot is None:
                continue
            fx, fy = spot
            heading = d + math.pi                    # someone coming up this corridor towards the junction

            def arrow_to(f):
                if not np.isfinite(f[cell]) or f[cell] < 2.5:
                    return None
                way = _downhill(plan, f, cell)
                if way is None:
                    return None
                rel = math.atan2(math.sin(math.atan2(way[1], way[0]) - heading),
                                 math.cos(math.atan2(way[1], way[0]) - heading))
                return "ahead" if abs(rel) < math.radians(50) else "left" if math.radians(50) <= rel <= math.radians(130) \
                    else "right" if -math.radians(130) <= rel <= -math.radians(50) else None

            rows, here = [], None
            if wing_fields:                          # the wing this junction is in: its nearest department's
                here = wings.get(min(fields, key=lambda g: fields[g][cell]))
                for wing, f in sorted(wing_fields.items(), key=lambda wf: wf[1][cell]):
                    arrow = arrow_to(f) if wing != here else None
                    if arrow and len(rows) < 2:
                        rows.append((wing, arrow))
            for g, f in sorted(fields.items(), key=lambda gf: gf[1][cell]):
                if len(rows) == MAX_ROWS:
                    break
                if here is not None and wings.get(g) not in (here, None):
                    continue                         # another wing's department: its wing is on the sign
                arrow = arrow_to(f)
                if arrow:
                    rows.append((g, arrow))
            if len({a for _, a in rows}) >= 2:       # a real choice: places in more than one direction
                faces.append(Face(0, "direction", fx, fy, KINDS["direction"]["z"], d, rows=rows))
    return faces


# ---------------------------------------------------------------------------------- pictures
def _font(size_px: int):
    from PIL import ImageFont
    return ImageFont.load_default(size=size_px)


def _fit(draw, text, max_w, size):
    while size > 8:
        font = _font(size)
        if draw.textlength(text, font=font) <= max_w:
            return font
        size -= 2
    return _font(size)


def tag_square(kind: str) -> tuple[float, float, float]:
    """Where a face's tag is, in metres from the face's top-left corner: (left, top, side). A directory
    has it in its top-left corner, the other signs halfway down their left end."""
    k = KINDS[kind]
    if kind == "directory":
        return TAG_MARGIN, TAG_MARGIN, k["tag"]
    return TAG_MARGIN, (k["h"] - k["tag"]) / 2, k["tag"]


def text_box(kind: str) -> tuple[float, float, float, float]:
    """The part of a face with the words, in metres from its top-left corner: (left, top, width, height).
    A directory is read whole (its header, then two columns of rows)."""
    k = KINDS[kind]
    if kind == "directory":
        return 0.0, 0.0, k["w"], k["h"]
    left = k["tag"] + 2 * TAG_MARGIN
    return left, 0.0, k["w"] - left, k["h"]


def directory_layout():
    """Where things are on a directory, in metres from its top-left corner: the header's height, a row's
    height, and the two columns' (left, right) edges. Each row has a name on the left of its column and
    the wing on the right."""
    k = KINDS["directory"]
    header = k["tag"] + 2 * TAG_MARGIN
    row_h = (k["h"] - header - 0.02) / DIRECTORY_ROWS
    return header, row_h, [(0.04, k["w"] / 2 - 0.03), (k["w"] / 2 + 0.03, k["w"] - 0.04)]


def arrow_points(direction: str, cx: float, cy: float, s: float):
    """An arrow's outline (pixels), pointing left, right or ahead (up)."""
    shape = [(-0.5, 0.0), (0.0, -0.5), (0.0, -0.18), (0.5, -0.18), (0.5, 0.18), (0.0, 0.18), (0.0, 0.5)]  # left
    pts = []
    for px_, py_ in shape:
        if direction == "right":
            px_ = -px_
        elif direction == "ahead":
            px_, py_ = py_, px_
        pts.append((cx + px_ * s, cy + py_ * s))
    return pts


def render(face: Face) -> np.ndarray:
    """The face's picture (BGR), PIXELS_PER_M pixels per metre."""
    from PIL import Image, ImageDraw
    k = KINDS[face.kind]
    W, H = int(round(k["w"] * PIXELS_PER_M)), int(round(k["h"] * PIXELS_PER_M))
    colour = COLOURS.get(face.colour, COLOURS["ward"]) if face.kind in ("plate", "blade") else (30, 40, 55)
    img = Image.new("RGB", (W, H), colour)
    draw = ImageDraw.Draw(img)
    left, top, side = tag_square(face.kind)
    box = side + 2 * TAG_MARGIN
    draw.rectangle([0, 0, int(box * PIXELS_PER_M), H if face.kind != "directory" else int(box * PIXELS_PER_M)],
                   fill=(255, 255, 255))                 # white behind the tag (a directory's is in its corner)
    tag_px = int(round(side * PIXELS_PER_M))
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO), face.tag, tag_px)
    img.paste(Image.fromarray(tag).convert("RGB"), (int(round(left * PIXELS_PER_M)), int(round(top * PIXELS_PER_M))))
    tx, ty, tw, th = [v * PIXELS_PER_M for v in text_box(face.kind)]
    pad = 0.04 * PIXELS_PER_M
    white = (255, 255, 255)
    if face.kind in ("plate", "blade"):
        words = face.place.split()
        size = int(th * (0.36 if not face.staff else 0.30))
        names = [face.place]
        if _fit(draw, face.place, tw - 2 * pad, size).size < 0.8 * size and len(words) > 1 and not face.staff:
            cut = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
            names = [" ".join(words[:cut]), " ".join(words[cut:])]
            size = int(th * 0.30)
        lines = names + (["STAFF ONLY"] if face.staff else [])
        name_font = min((_fit(draw, n, tw - 2 * pad, size) for n in names), key=lambda f: f.size)
        fonts = [name_font] * len(names) + ([_font(int(size * 0.62))] if face.staff else [])
        heights = [f.size for f in fonts]
        y = ty + (th - sum(heights) - 0.25 * heights[0] * (len(lines) - 1)) / 2
        for line, f in zip(lines, fonts):
            draw.text((tx + pad, y), line, font=f, fill=white)
            y += f.size * 1.25
    elif face.kind == "direction":
        n = max(1, len(face.rows))
        row_h = th / MAX_ROWS
        y0 = ty + (th - row_h * n) / 2
        size = int(row_h * 0.55)
        for i, (group, arrow) in enumerate(face.rows):
            cy = y0 + row_h * (i + 0.5)
            arrow_x = tx + pad + row_h * 0.35
            draw.polygon(arrow_points(arrow, arrow_x, cy, row_h * 0.62), fill=white)
            font = _fit(draw, group, tw - 2 * pad - row_h * 0.9, size)
            draw.text((arrow_x + row_h * 0.55, cy - font.size * 0.55), group, font=font, fill=white)
            if i < n - 1:
                draw.line([tx + pad, y0 + row_h * (i + 1), tx + tw - pad, y0 + row_h * (i + 1)], fill=(120, 130, 145), width=2)
    elif face.kind == "directory":
        header, row_h, cols = [v * PIXELS_PER_M if not isinstance(v, list) else v for v in directory_layout()]
        grey = (190, 205, 225)
        title = _font(int(header * 0.42))
        draw.text((box * PIXELS_PER_M + pad, header * 0.10), "Directory", font=title, fill=white)
        if face.here:
            here_font = _fit(draw, f"You are here: {face.here}", W / 2 - 2 * pad, int(header * 0.30))
            draw.text((W / 2, header * 0.58), f"You are here: {face.here}", font=here_font, fill=white)
        draw.line([pad, header, W - pad, header], fill=(120, 130, 145), width=3)
        for i, (name, wing) in enumerate(face.rows):
            col, row = divmod(i, DIRECTORY_ROWS)
            x0, x1 = [v * PIXELS_PER_M for v in cols[col]]
            cy = header + row_h * (row + 0.5)
            wing_font = _font(int(row_h * 0.48))
            wing_w = draw.textlength(wing, font=wing_font) if wing else 0.0
            if wing:
                draw.text((x1 - wing_w, cy - wing_font.size * 0.55), wing, font=wing_font, fill=grey)
            name_font = _fit(draw, name, x1 - x0 - wing_w - 0.05 * PIXELS_PER_M, int(row_h * 0.56))
            draw.text((x0, cy - name_font.size * 0.6), name, font=name_font, fill=white)
    out = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    pow2 = lambda v: 1 << (int(v) - 1).bit_length()       # Webots wants texture sides that are powers of two
    return cv2.resize(out, (pow2(W), pow2(H)), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------------- cache and world
def _plan_key(plan, world: str) -> str:
    """Changes when the floor plan or this file changes, so the signs are drawn again."""
    h = hashlib.sha1(plan.cells.tobytes() + Path(__file__).read_bytes()).hexdigest()[:16]
    return f"{world}-{h}"


def load(plan, world: str) -> Layout:
    """A world's signs: named as its building (fp_hospital_v2 as fp_hospital), hung to fit its own
    floor plan (its furniture is elsewhere), drawn and cached in webots/worlds/signs/<world>/ the first
    time, then read from there."""
    folder = TEXTURES / world
    meta = folder / "signs.json"
    key = _plan_key(plan, world)
    if meta.exists():
        data = json.loads(meta.read_text())
        if data.get("key") == key:
            faces = [Face(**{**f, "rows": [tuple(r) for r in f["rows"]], "stop": tuple(f["stop"])}) for f in data["faces"]]
            return Layout(world, faces, [tuple(d) for d in data["doors"]], [tuple(b) for b in data["boards"]],
                          {k: tuple(v) for k, v in data["stops"].items()})
    layout = make_layout(plan, building_of(world))
    layout.building = world
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    for f in layout.faces:
        cv2.imwrite(str(folder / f"{f.tag}.png"), render(f))
    meta.write_text(json.dumps({"key": key, "faces": [asdict(f) for f in layout.faces], "doors": layout.doors,
                                "boards": layout.boards, "stops": layout.stops}, indent=1))
    return layout


def building_of(world_stem: str) -> str:
    return re.sub(r"_(v\d+|test)$", "", world_stem)


def _face_node(f: Face, building: str) -> str:
    k = KINDS[f.kind]
    w, h = k["w"] / 2, k["h"] / 2
    url = f"signs/{building}/{f.tag}.png"
    tex = f'ImageTexture {{ url ["{url}"] repeatS FALSE repeatT FALSE }}'
    return (f"Pose {{ translation {f.x:.4f} {f.y:.4f} {f.z:.4f} rotation 0 0 1 {f.yaw:.5f} children [ Shape {{ "
            f"appearance PBRAppearance {{ baseColorMap {tex} emissiveColorMap {tex} emissiveIntensity 0.45 "
            f"roughness 1 metalness 0 }} geometry IndexedFaceSet {{ coord Coordinate {{ point [ "
            f"0.001 {-w:.4f} {-h:.4f}, 0.001 {w:.4f} {-h:.4f}, 0.001 {w:.4f} {h:.4f}, 0.001 {-w:.4f} {h:.4f} ] }} "
            f"texCoord TextureCoordinate {{ point [ 0 0, 1 0, 1 1, 0 1 ] }} coordIndex [ 0 1 2 3 -1 ] "
            f"texCoordIndex [ 0 1 2 3 -1 ] }} }} ] }}")


def _board_node(x, y, z, yaw, w, h) -> str:
    return (f"Pose {{ translation {x:.4f} {y:.4f} {z:.4f} rotation 0 0 1 {yaw:.5f} children [ Shape {{ "
            f"appearance PBRAppearance {{ baseColor 0.82 0.82 0.80 roughness 1 metalness 0 }} "
            f"geometry Box {{ size 0.01 {w + 0.012:.3f} {h + 0.012:.3f} }} }} ] }}")


def _door_node(x, y, yaw) -> str:
    return (f"Pose {{ translation {x:.4f} {y:.4f} {DOOR_H / 2:.3f} rotation 0 0 1 {yaw:.5f} children [ "
            f"Shape {{ appearance PBRAppearance {{ baseColor 0.55 0.42 0.30 roughness 0.8 metalness 0 }} "
            f"geometry Box {{ size 0.03 {DOOR_W} {DOOR_H} }} }} "
            f"Pose {{ translation 0.03 {DOOR_W / 2 - 0.1:.3f} 0 children [ Shape {{ appearance PBRAppearance {{ "
            f"baseColor 0.75 0.75 0.75 roughness 0.3 metalness 0.8 }} geometry Box {{ size 0.04 0.12 0.03 }} }} ] }} ] }}")


def add_to_world(sup, layout: Layout) -> None:
    """Hang the signs (and draw the doors) in the running world, once."""
    if not layout.faces or sup.getFromDef("SIGNS") is not None:
        return
    nodes = [_face_node(f, layout.building) for f in layout.faces]
    nodes += [_board_node(*b) for b in layout.boards]
    nodes += [_door_node(*d) for d in layout.doors]
    sup.getRoot().getField("children").importMFNodeFromString(-1, "DEF SIGNS Group { children [\n" + "\n".join(nodes) + "\n] }")


def for_world(sup, plan, world_stem: str) -> Layout:
    layout = load(plan, world_stem)
    add_to_world(sup, layout)
    return layout


# ---------------------------------------------------------------------------------- a look at a layout
def draw_map(plan, layout: Layout, path: Path, scale: float = 2.0) -> None:
    """A picture of the floor plan with every sign: plates and blades as coloured dots with their
    names, direction signs as white triangles, drawn doors as brown bars."""
    img = np.full(plan.cells.shape + (3,), 255, np.uint8)
    img[plan.cells == WALL] = (0, 0, 0)
    img[plan.cells == UNKNOWN] = (190, 190, 190)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    def p(x, y):
        r, c = plan._pixel(x, y)
        return int(c * scale), int(r * scale)
    for x, y, yaw in layout.doors:
        a, b = p(x - math.sin(yaw) * DOOR_W / 2, y + math.cos(yaw) * DOOR_W / 2), \
            p(x + math.sin(yaw) * DOOR_W / 2, y - math.cos(yaw) * DOOR_W / 2)
        cv2.line(img, a, b, (40, 90, 150), 4)
    for f in layout.faces:
        colour = COLOURS.get(f.colour, (80, 80, 80))[::-1]
        if f.kind == "plate":
            cv2.circle(img, p(f.x, f.y), 5, colour, -1)
            cv2.putText(img, f.place, (p(f.x, f.y)[0] + 6, p(f.x, f.y)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)
        elif f.kind == "directory":
            cv2.rectangle(img, (p(f.x, f.y)[0] - 7, p(f.x, f.y)[1] - 7), (p(f.x, f.y)[0] + 7, p(f.x, f.y)[1] + 7), (0, 150, 0), -1)
            cv2.putText(img, f"directory ({f.here})" if f.here else "directory", (p(f.x, f.y)[0] + 9, p(f.x, f.y)[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 120, 0), 1)
        elif f.kind == "direction":
            c = np.array(p(f.x, f.y))
            d = np.array([math.cos(f.yaw), -math.sin(f.yaw)])
            n = np.array([-d[1], d[0]])
            tri = np.array([c + d * 9, c - d * 5 + n * 6, c - d * 5 - n * 6], np.int32)
            cv2.fillPoly(img, [tri], (0, 0, 220))
    cv2.imwrite(str(path), img)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Print a building's signs and draw where they are.")
    p.add_argument("world", help="corridor, fp_hospital, fp_office, ...")
    p.add_argument("--picture", help="also save a picture of the floor plan with the signs on it, for checking")
    args = p.parse_args(argv)
    settings = json.loads((WEBOTS_DIR / "worlds" / f"{args.world}.json").read_text())
    plan = load_floorplan(str((ROOT / settings["map"]).resolve()))
    layout = load(plan, args.world)
    for f in layout.faces:
        print(f"tag {f.tag:3d}  {f.kind:9s}  ({f.x:6.1f}, {f.y:6.1f}, {f.z:.2f})  {f.text}")
    print(f"{len(layout.places())} places, {len(layout.faces)} sign faces, {len(layout.doors)} doors drawn")
    if args.picture and layout.faces:
        draw_map(plan, layout, Path(args.picture))
        print(f"Picture: {args.picture}")


if __name__ == "__main__":
    main()
