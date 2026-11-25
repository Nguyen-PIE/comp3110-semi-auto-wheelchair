"""Watch the chair drive in Webots, take control, or measure it over many trips.

    python webots/run.py corridor                      # opens Webots; the learned assistant drives
    python webots/run.py fp_hospital --driver rule     # any driver from the 2D simulator
    python webots/run.py corridor --episodes 20 --fast # 20 trips as fast as possible, then a summary
    python webots/run.py corridor --attach             # use a Webots you already opened on that world
    python webots/run.py fp_hospital --landmarks       # new coloured markers to head for, each time O is pressed
    python webots/run.py fp_hospital --landmarks --low-obstacles --sensors   # coffee tables, and what the sensors see
    python webots/run.py fp_hospital --phones          # up to 4 phones walk people round the chair (phones.py)
    python webots/run.py fp_hospital --signs           # with the room and direction signs
    python webots/run.py fp_hospital --open-world      # no floor plan: finds rooms by their signs, markers to aim at (O)
    python webots/run.py fp_hospital --open-world --landmarks 0   # the same without the markers and red line
    (voice commands are always on: hold K and say what to do, "I need the pharmacy", see voice.py)

The chair leaves a blue line on the floor where it has driven (--no-trail hides it).
Make the world first with webots/make_world.py. In the Webots window, click the 3D view, then use
the same keys as the 2D demo (scripts/demo.py); L lists them at the bottom of the 3D view:
    O       take control / hand back
    Arrows  drive while you have control (Up forward, Left and Right turn; no keys = stop)
    Space   pause / resume (the trip clock stops too)
    M       camera: following the chair / the trip, as in the 2D demo / the whole building (V too)
    F       first-person view: from the rider's eye height, then through the eyes of each person
            walked from a phone (--phones)
    S       show / hide what the chair's sensors can see (a fan on the floor, rings round people it sees);
            on by itself while the rider drives (O), S hides it for that time
    Z       show / hide the proxemic zones round the chair (assist/proxemics.py), also --zones.
            The zones and the sensor fan are both see-through layers on the floor, and where they
            overlap Webots draws them in a changing order (the colours flicker), so only one shows at a
            time: Z hides the fan, S hides the zones, and the fan does not come on by itself over them
    L       show / hide the list of keys
    H       hide / show the status lines and lists at the top (a clear first-person view)
    R       restart the trip with a new seed (--open-world: the same seed, the same start and people)
    N       next building (opens it in a new Webots, the camera back on its starting view)
    Q       quit (Esc too, if Webots passes it on)
    K       hold it, say what to do (stop, hand back, I need the pharmacy...), let go
Every trip prints one line; at the end you get the same summary as eval/run_episodes.py.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

WEBOTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEBOTS_DIR.parent))

from webots import launch  # noqa: E402

VALIDATION_SEEDS = range(10_000, 10_200)   # as envs/make_env.py: never used for training


def world_path(name: str) -> Path:
    path = Path(name)
    if path.suffix != ".wbt":
        path = WEBOTS_DIR / "worlds" / f"{name}.wbt"
    if not path.exists():
        sys.exit(f"{path} not found. Make it first:  python webots/make_world.py {path.stem}")
    return path


VIEW_KEYS = {ord("M"), ord("V")}   # M as in the 2D demo; V also works
FIRST_PERSON_KEY = ord("F")
SENSOR_KEY = ord("S")
TRACKS_KEY = ord("T")
ZONES_KEY = ord("Z")
TEXT_KEY = ord("H")                # hides the status lines and lists, for a clear view
QUIT_KEYS = {ord("Q"), 27}         # 27 is Esc, if Webots passes it on
KEY_HELP = [
    "O       take control / hand back (Space pauses)",
    "Arrows  drive while you have control (no keys = stop)",
    "M       camera: around the chair / trip (as in 2D) / whole building",
    "F       first person: the rider's eye, then each phone walker's, then back",
    "S / Z   what the sensors see / the proxemic zones round the chair (one at a time)",
    "T       people's tracks on / off: orange predicted, blue walked, black real",
    "        rings: green walking, magenta phone, blue group, yellow stopped, purple older",
    "Arrows  in manual mode: aim the red line; a marker it points at locks on",
    "1-8 G   pick a marker by number; G drives to the locked or picked marker, or the red post",
    "L H R N Q  list / text at the top / restart trip / next building / quit",
]
GREEN, ORANGE, GREY, INK, RED, WHITE = 0x006E28, 0xC04B00, 0x3C3C3C, 0x000000, 0xC00000, 0xFFFFFF
OUTLINE = [(-0.0015, 0.0), (0.0015, 0.0), (0.0, -0.002), (0.0, 0.002)]   # white copies around each text
TEXT_EVERY_S = 0.25         # a line of text that changes all the time (speed, distances) goes to Webots at most this often
# Webots numbers the labels over the 3D view and takes only 0 to 99: the status lines, their
# outlines, the list of keys and its outlines each get their own block of numbers.
STATUS_ROWS, STATUS_OUTLINES, KEY_LABELS, KEY_OUTLINES = 6, 6, 30, 40
RELIST_M, RELIST_RAD = 1.0, math.radians(20)   # the open world: which markers are in sight, again after this move or turn
LEGEND_ROWS, LEGEND_LABELS, LEGEND_OUTLINES, LEGEND_PER_ROW = 4, 80, 84, 3     # 3 rows of landmarks, 1 of aim
assert STATUS_OUTLINES + 4 * STATUS_ROWS <= KEY_LABELS and KEY_LABELS + len(KEY_HELP) <= KEY_OUTLINES
assert LEGEND_OUTLINES + 4 * LEGEND_ROWS <= 100 and KEY_OUTLINES + 4 * len(KEY_HELP) <= LEGEND_LABELS


VIEWS = ["chair", "trip", "whole", "first"]   # M cycles through them in this order; the first is the default
OVERVIEWS = VIEWS[:3]                    # the views from above; M skips the first-person one
TRIP_VIEW_WIDTH_M = 33.0   # as the 2D demo's floor plans: a 1000-pixel view at 30 pixels per metre
VIEW_MARGIN_M = 3.0        # as the 2D demo: kept between the chair and the edge of the view
SCREEN_ASPECT = 1.6        # the 3D view's width / height, to keep the chair in sight up and down
OVERVIEW_FIELD_OF_VIEW = 0.785   # the worlds' Viewpoint fieldOfView, which Webots measures across
HALF_FIELD_OF_VIEW = OVERVIEW_FIELD_OF_VIEW / 2
FIRST_PERSON_FIELD_OF_VIEW = 1.2         # as the chair's forward camera in protos/Wheelchair.proto
EYE_HEIGHT_M = 1.15                      # the chair camera's height: about a seated rider's eyes
EYE_FORWARD_M = 0.1                      # and how far in front of the chair's centre it sits
WALKER_EYE_HEIGHT_M = 1.6                # a walked person's eyes (players.py) ...
WALKER_EYE_FORWARD_M = 0.1               # ... at the front of their head: never inside a wall or the chair beside them
TILT = 0.2                 # how far back the camera sits from what it looks at, per metre of height
CHAIR_VIEW_WIDTH_M = 20.0  # the chair view: this much of the map around the chair, across the screen


class Views:
    """The camera, which M switches between three views from steeply above, so that walls do
    not hide the chair, and F between the first-person view and the last overview:
        chair   an overview of the part of the map around the chair (20 m across), centred on the
                chair and moving with it exactly (Webots' smoothed following drifts off whenever the
                chair jumps to a new start); the default, on every map. Scroll to zoom in or out
        trip    as the 2D demo's floor plans: a fixed zoom (33 m across, or the whole building if it
                is smaller), centred on the chair's start, goal and route; when they do not all fit,
                the view slides just enough to keep the chair 3 m inside it
        whole   the whole building, its long side across the screen
        first   the rider's view: from the chair's camera position, looking where the chair points,
                so walls hide what is behind them, as they do for the chair's own sensors; or, after
                more presses of F, through the eyes of each person walked from a phone"""

    def __init__(self, scene, view: str):
        from envs.floorplan import ROUTE_CELL_M
        from webots.make_world import look_at
        self.scene, self.view, self.look_at = scene, view, look_at
        children = scene.sup.getRoot().getField("children")
        nodes = [children.getMFNode(i) for i in range(children.getCount())]
        self.viewpoint = next((n for n in nodes if n.getTypeName() == "Viewpoint"), None)
        plan, cells = scene.plan, scene.plan.spawn_cells     # where people can be: the building's floor
        xs = (cells[:, 1] + 0.5) * ROUTE_CELL_M - plan.width_m / 2
        ys = plan.height_m / 2 - (cells[:, 0] + 0.5) * ROUTE_CELL_M
        self.floor = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
        x0, x1, y0, y1 = self.floor
        # The trip view's width: the 2D demo's, or less if the whole building fits in less.
        self.width = min(TRIP_VIEW_WIDTH_M, max(x1 - x0 + 2.0, (y1 - y0 + 2.0) * SCREEN_ASPECT))
        self.anchor = None                    # the trip view's centre before sliding to keep the chair
        self.overview = view if view in OVERVIEWS else OVERVIEWS[0]   # what F returns to
        self.eyes = None                      # first person: None for the rider, or a player's number

    def _place(self, eye, target) -> None:
        field = self.viewpoint.getField
        field("position").setSFVec3f([float(v) for v in eye])
        field("orientation").setSFRotation([float(v) for v in self.look_at(eye, target).split()])

    def show(self, view: str | None = None) -> None:
        """Switch to `view`, or set the current one up again for a new trip."""
        if self.viewpoint is None:
            return
        self.view = view or self.view
        if self.view in OVERVIEWS:
            self.overview = self.view
        self.viewpoint.getField("follow").setSFString("")     # let go of the chair while moving
        if hasattr(self.scene, "rider_on_view"):            # first person looks out of the rider's eyes
            self.scene.rider_on_view(self.viewpoint, not (self.view == "first" and self.eyes is None))
        self.viewpoint.getField("fieldOfView").setSFFloat(
            FIRST_PERSON_FIELD_OF_VIEW if self.view == "first" else OVERVIEW_FIELD_OF_VIEW)
        if self.view == "first":
            self.update()
        elif self.view == "trip":
            r = self.scene.robot
            points = [(r.x, r.y), (r.goal_x, r.goal_y)] + [tuple(p) for p in getattr(self.scene, "route", [])]
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            self.anchor = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
            self.update()
        elif self.view == "chair":
            x, y = self.scene.robot.x, self.scene.robot.y
            height = self._height_for(CHAIR_VIEW_WIDTH_M)
            self._place((x, y - TILT * height, height), (x, y, 0.0))
            self.viewpoint.getField("followSmoothness").setSFFloat(0.0)
            self.viewpoint.getField("follow").setSFString("wheelchair")
        else:
            x0, x1, y0, y1 = self.floor
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            height = max(1.3 * max(w, h), 2.0 * min(w, h), 10.0)   # fits a 45-degree view
            eye = (cx, cy - 0.3 * height, height) if w >= h else (cx - 0.3 * height, cy, height)
            self._place(eye, (cx, cy, 0.0))

    def update(self) -> None:
        """Every step, in the trip view: as the 2D demo's frame_box, the view stays centred on the
        trip and slides only as far as needed to keep the chair 3 m inside it. It never zooms."""
        if self.viewpoint is not None and self.view == "first":
            who = self.scene.robot if self.eyes is None else self.scene.crowd.player(self.eyes)
            if who is None:                       # that person was let go, or a new trip began
                self.eyes = None
                self.show(self.overview)
                return
            forward, height = (EYE_FORWARD_M, EYE_HEIGHT_M) if self.eyes is None else \
                (WALKER_EYE_FORWARD_M, WALKER_EYE_HEIGHT_M)
            ahead = (math.cos(who.orientation), math.sin(who.orientation))
            eye = (who.x + forward * ahead[0], who.y + forward * ahead[1], height)
            self._place(eye, (eye[0] + ahead[0], eye[1] + ahead[1], height))
            return
        if self.viewpoint is None or self.view != "trip" or self.anchor is None:
            return
        r = self.scene.robot
        half_w = self.width / 2
        half_h = half_w / SCREEN_ASPECT
        keep_x, keep_y = min(VIEW_MARGIN_M, half_w), min(VIEW_MARGIN_M, half_h)
        cx = float(np.clip(self.anchor[0], r.x - half_w + keep_x, r.x + half_w - keep_x))
        cy = float(np.clip(self.anchor[1], r.y - half_h + keep_y, r.y + half_h - keep_y))
        height = self._height_for(self.width)
        self._place((cx, cy - TILT * height, height), (cx, cy, 0.0))

    @staticmethod
    def _height_for(width: float) -> float:
        """How high the camera must be to show `width` metres of floor across the screen."""
        return width / 2 / math.tan(HALF_FIELD_OF_VIEW) / math.hypot(1.0, TILT)

    def toggle(self) -> None:
        """M: the next overview (from the first-person view, the last overview it left)."""
        current = self.overview if self.view == "first" else self.view
        self.show(OVERVIEWS[(OVERVIEWS.index(current) + (self.view != "first")) % len(OVERVIEWS)])

    def toggle_first_person(self) -> None:
        """F: the rider's eyes, then the eyes of each person walked from a phone, then back to the
        overview it came from."""
        from webots.players import SLOTS
        options = [None] + [c.number for c in SLOTS if self.scene.crowd.player(c.number) is not None]
        if self.view != "first":
            self.eyes = None
            self.show("first")
            return
        following = options.index(self.eyes) + 1 if self.eyes in options else len(options)
        if following < len(options):
            self.eyes = options[following]
            self.show("first")
        else:
            self.eyes = None
            self.show(self.overview)


class Overlay:
    """Text over the 3D view, like the lines under the map in the 2D demo: who is driving, the
    trip, what the assistant is doing, the last trip and the phones walking people at the top; L
    shows the keys at the bottom. H hides the lines at the top and the list under them.
    Black bold-ish text with a thin white outline, so it reads on the light floor and on anything
    dark behind it."""

    def __init__(self, sup, help_lines: list[str] = KEY_HELP):
        self.sup, self.keys_shown, self.help = sup, False, help_lines   # the list of keys is hidden until L
        self.text_shown = True                        # the status lines and the list under them, until H
        self.last_legend = None                       # what legend() last drew, to draw again when H is pressed
        self.drawn: dict[int, tuple] = {}             # label number: what it shows, so unchanged text is not sent again
        self.sent: dict[int, float] = {}              # label number: when it was last sent
        self.pending: dict[int, tuple] = {}           # label number: newer text, sent once TEXT_EVERY_S has passed
        assert len(help_lines) <= len(KEY_HELP)       # it has as many labels as run.py's own list

    def _text(self, slot: int, outline: int, text: str, x: float, y: float, size: float, colour: int,
              font: str) -> None:
        """One line of text at label `slot`, with its white outline at labels outline to outline + 3.
        Webots draws a label with a lower number over one with a higher number, so the outlines go
        on higher numbers than the text. A label already showing this is left alone: every step
        redraws the lines, and sending 50 unchanged labels each time costs Webots and Python for nothing.
        A line that changes every step (the speed, how far to go) is sent at most every TEXT_EVERY_S:
        Webots makes Python wait for each label, 5 a line, and that made a moving chair slow."""
        want = (text, x, y, size, colour, font)
        if self.drawn.get(slot) == want:
            self.pending.pop(slot, None)
            return
        if time.monotonic() - self.sent.get(slot, -math.inf) < TEXT_EVERY_S:
            self.pending[slot] = (outline, want)       # flush() sends it a little later
            return
        self._send(slot, outline, want)

    def _send(self, slot: int, outline: int, want: tuple) -> None:
        text, x, y, size, colour, font = want
        self.drawn[slot], self.sent[slot] = want, time.monotonic()
        self.pending.pop(slot, None)
        for k, (dx, dy) in enumerate(OUTLINE):
            self.sup.setLabel(outline + k, text, x + dx, y + dy, size, WHITE, 0.0, font)
        self.sup.setLabel(slot, text, x, y, size, colour, 0.0, font)

    def flush(self) -> None:
        """Sends the text held back by _text once it may be (lines() calls it every step)."""
        now = time.monotonic()
        for slot, (outline, want) in list(self.pending.items()):
            if now - self.sent.get(slot, -math.inf) >= TEXT_EVERY_S:
                self._send(slot, outline, want)

    def lines(self, rows: list[tuple[str, int]]) -> None:
        self.flush()
        y = 0.012
        for i, (text, colour) in enumerate(rows[:STATUS_ROWS]):
            size = 0.05 if i == 0 else 0.045               # all in Webots' bold font, Arial Black
            self._text(i, STATUS_OUTLINES + 4 * i, text if self.text_shown else "", 0.012, y, size, colour,
                       "Arial Black")
            y += size + 0.006

    def show_keys(self, shown: bool) -> None:
        self.keys_shown = shown
        for i, text in enumerate(self.help):            # a fixed-width font keeps the keys in a column
            self._text(KEY_LABELS + i, KEY_OUTLINES + 4 * i, text if shown else "", 0.012, 0.60 + 0.042 * i,
                       0.036, INK, "Lucida Console")

    def legend(self, names: list[str], selected: int | None = None, aim: str = "", title: str = "Landmarks") -> None:
        """The landmarks, numbered for the selector keys, a few to a row, under the status lines.
        The selected one is in [brackets]. The last row says what the chair is aiming at."""
        self.last_legend = (names, selected, aim, title)
        items = [f"[{i + 1} {n}]" if i == selected else f"{i + 1} {n}" for i, n in enumerate(names)]
        rows = [items[i:i + LEGEND_PER_ROW] for i in range(0, len(items), LEGEND_PER_ROW)][:LEGEND_ROWS - 1]
        for i in range(LEGEND_ROWS):
            if i == LEGEND_ROWS - 1:
                text = aim
            else:
                text = (f"{title}: " if i == 0 else " " * (len(title) + 2)) + ", ".join(rows[i]) if i < len(rows) else ""
            self._text(LEGEND_LABELS + i, LEGEND_OUTLINES + 4 * i, text if self.text_shown else "", 0.012,
                       0.35 + 0.042 * i, 0.036, INK, "Lucida Console")


def driver_name(driver) -> str:
    """Which assistant is driving, for the screen: the one trained with reinforcement learning or not."""
    kind = type(driver).__name__
    return {"AssistDriver": "Learned assistant (RL)", "RuleDriver": "Rule-based assistant"}.get(kind, "Assistant")


def assistant_text(driver, manual: bool) -> tuple[str, int]:
    """What the assistant is doing, as in the 2D demo's line under the map, and which one it is."""
    name = driver_name(driver)
    try:
        risky = sum(t.collision_predicted for t in driver.tracks)
        if manual:
            text = f"{name}: paused while you drive"
        else:
            speed, offset = driver.adjustment
            degrees = math.degrees(offset)
            if not driver.controller.engaged and abs(degrees) < 2:
                text = f"{name}: route clear, following it"
            else:
                steer = "on the route" if abs(degrees) < 2 else f"{abs(degrees):.0f} deg {'left' if degrees > 0 else 'right'}"
                text = f"{name}: speed {speed:.0%}, steering {steer}"
            if driver.controller.safety_stop:
                text += "  SAFETY STOP"
            if getattr(driver.controller, "asking", False):
                text += "  Excuse me: asking people to make way"
    except AttributeError:                       # a driver without the learned assistant's state
        return "", INK
    if risky:
        text += f"  ({risky} {'person' if risky == 1 else 'people'} too close)"
    return text, RED if risky else INK


def new_landmarks(marks, scene, overlay, seed: int) -> None:
    """Hang new landmarks where the chair is now, and list them."""
    r = scene.robot
    items = marks.refresh(r.x, r.y, r.orientation, seed)
    print("New landmarks: " + ", ".join(f"{i + 1} {lm.name}" for i, lm in enumerate(items)), flush=True)
    if overlay is not None:
        overlay.legend([lm.name for lm in items])


def list_markers(marks, building, scene) -> None:
    """The open world: all its markers are up all the time (open_world.BuildingMarkers). The ones in
    sight of the chair now, up to 9 and numbered left to right, are those the number keys, the red
    line and voice pick from. A marker picked stays picked while it is still in sight."""
    r, picked = scene.robot, marks.selector.selected
    shown = building.in_sight(r.x, r.y, r.orientation)
    if [lm.name for lm in shown] == [lm.name for lm in marks.items]:
        return
    marks.items = marks.selector.landmarks = shown
    names = [lm.name for lm in shown]
    marks.selector.select(names.index(picked.name) if picked is not None and picked.name in names else None)


def next_building(world: Path) -> str:
    from webots.splits import HELD_OUT_BUILDINGS, TRAIN_BUILDINGS
    buildings = TRAIN_BUILDINGS + HELD_OUT_BUILDINGS
    base = re.sub(r"_(v\d+|test)$", "", world.stem)
    return buildings[(buildings.index(base) + 1) % len(buildings)] if base in buildings else buildings[0]


def hide_camera_preview(world: Path, chair_camera: bool = True) -> None:
    """Webots draws each camera on the chair as a black box over the 3D view, even while it is off.
    Hide them in the world's Webots layout file (.<world>.wbproj, which Webots keeps beside the
    world and reads when it opens it): the phones' walker eyes always, the chair's own camera
    unless --camera asks to see it."""
    path = world.parent / f".{world.stem}.wbproj"
    devices = (["camera"] if chair_camera else []) + [f"walker eye {n}" for n in range(1, 5)]
    try:
        lines = path.read_text().splitlines() if path.exists() else ["Webots Project File version R2025a"]
        lines = [line for line in lines if not line.startswith("renderingDevicePerspectives:")]
        path.write_text("\n".join(lines + [f"renderingDevicePerspectives: wheelchair:{d};0;1;0;0" for d in devices]) + "\n")
    except OSError:
        pass                                    # only the look of the window is at stake


def manual_command(keys: set[int], Keyboard) -> np.ndarray:
    forward = 1.0 if Keyboard.UP in keys else 0.0
    turn = (1.0 if Keyboard.LEFT in keys else 0.0) - (1.0 if Keyboard.RIGHT in keys else 0.0)
    return np.array([forward, 0.0, turn], dtype=np.float32)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("world", help="a world made by make_world.py (corridor, fp_hospital, ...) or a .wbt path")
    p.add_argument("--driver", default="auto", help="auto, assist, rule or route_follower (default auto)")
    p.add_argument("--episodes", type=int, default=0, help="stop after this many trips (default: run until closed)")
    p.add_argument("--seed", type=int, default=VALIDATION_SEEDS.start, help="first scene seed (default 10000)")
    p.add_argument("--walls", choices=["lidar", "map"], default="lidar",
                   help="where the assistant's wall distances come from (default lidar)")
    p.add_argument("--fast", action="store_true", help="run as fast as possible without the 3D view")
    p.add_argument("--headless", action="store_true", help="no screen needed (Linux servers, uses xvfb-run)")
    p.add_argument("--attach", action="store_true", help="connect to a Webots that is already open")
    p.add_argument("--port", type=int, default=1234)
    p.add_argument("--model", help="the policy for the assist driver (default models/assist_ppo.zip)")
    p.add_argument("--camera", action="store_true", help="also switch on the chair's forward camera")
    p.add_argument("--no-trail", action="store_true", help="do not draw the line the chair has driven")
    p.add_argument("--no-brake", action="store_true",
                   help="while the rider drives (O): only warn about people in the way, do not slow down for them")
    p.add_argument("--landmarks", type=int, nargs="?", const=6, default=None, metavar="N",
                   help="hang N coloured landmarks along the walls, new ones around the chair each time O is pressed (default 6) "
                        "for the rider to refer to, such as the yellow sphere, with the red aiming line while the rider "
                        "drives. On by default with --open-world, shown there only while the rider drives "
                        "(--landmarks 0 switches them off)")
    p.add_argument("--low-obstacles", type=int, nargs="?", const=4, default=4, metavar="N",
                   help="N low obstacles (coffee tables, poufs, tables) on the floor each trip: a rider looks over them at the "
                        "markers behind, but the chair's lidar sees them and the chair has to go round "
                        "(default 4, always on; --low-obstacles 0 turns them off)")
    p.add_argument("--signs", action="store_true",
                   help="hang the building's room and direction signs (signs.py), as in open_world.py")
    p.add_argument("--tracks", action="store_true",
                   help="start with each person's predicted walking track and kind of walker drawn (T switches it on and off)")
    p.add_argument("--zones", action="store_true",
                   help="start with the proxemic zones drawn round the chair (assist/proxemics.py; Z switches them "
                        "on and off): intimate, personal and social, the front stretching with speed")
    p.add_argument("--replay", nargs="+", metavar=("TRACKS.csv", "N"),
                   help="replace N of each trip's walkers (default 6) with recorded tracks from a CSV of id,t,x,y, such as "
                        "the SiT dataset made by scripts/sit_tracks.py (see webots/replay.py)")
    p.add_argument("--sensors", action="store_true",
                   help="start with what the chair's sensors can see drawn on the floor (S switches it on and off)")
    p.add_argument("--walkers", choices=["rules", "learned"], default="rules",
                   help="how people step: the 2D simulator's rules (default), or Trajectron++ trained on real "
                        "pedestrian recordings (webots/learned_walk.py)")
    p.add_argument("--crowd-scale", type=float, default=1.0, metavar="X",
                   help="this many times the world's usual number of walking people, at the start and kept up "
                        "around the chair all trip (default 1, the usual crowd; narrow places never get more than "
                        "0.05 people per square metre)")
    p.add_argument("--no-stream", action="store_true",
                   help="people only start near the first route: no new people appear around the chair or "
                        "along a new route after G, and none disappear when far away (as in training and testing)")
    p.add_argument("--phones", action="store_true",
                   help="let up to 4 phones on the same Wi-Fi walk people round the chair (webots/phones.py)")
    p.add_argument("--phones-port", type=int, default=8765, help="the phones' web address port (default 8765)")
    p.add_argument("--phones-code", help=argparse.SUPPRESS)      # kept when N opens the next building
    p.add_argument("--open-world", action="store_true",
                   help="no floor plan: the chair finds the places in the building's directory by reading their "
                        "signs (open_world.py); everything else is as in an ordinary trip")
    p.add_argument("--target", help="with --open-world: a place to look for straight away, such as \"CT Scan\"")
    p.add_argument("--voice", action="store_true", help=argparse.SUPPRESS)   # voice is always on now: kept for old commands
    p.add_argument("--voice-model", help="the gateway model that works out the rider's own spoken words "
                                         "(default COMP3110_MODEL_NAME in .env)")
    p.add_argument("--read", choices=["text", "tags"], default="text",
                   help="with --open-world: read sign names with OCR (text, default) or from the sign tags (tags)")
    p.add_argument("--view", choices=VIEWS, default="chair",
                   help="start the camera following the chair (chair), on the trip as in the 2D demo (trip), "
                        "on the whole building (whole) or through the chair's eyes (first); "
                        "M switches between the first three, F to the first-person view (default chair)")
    args = p.parse_args(argv)
    if args.landmarks is None:                  # the open world has the markers and the red line unless switched off
        args.landmarks = 8 if args.open_world else 0

    world = world_path(args.world)
    process = None
    if not args.attach:
        hide_camera_preview(world, chair_camera=not args.camera)
    if not args.attach:
        process = launch.start_webots(world, args.port, fast=args.fast or args.headless, headless=args.headless,
                                      log=WEBOTS_DIR / "webots.log")
    launch.connect_to(args.port)
    try:
        follow_up = run(world, args)
    finally:
        launch.stop_webots(process)
    if follow_up:                               # N: start again on the next building
        name, seed, view = follow_up
        again = [name, "--seed", str(seed), "--view", view, "--driver", args.driver, "--walls", args.walls,
                 "--port", str(args.port)]
        again += ["--model", args.model] if args.model else []
        again += ["--camera"] if args.camera else []
        again += ["--no-trail"] if args.no_trail else []
        again += ["--no-brake"] if args.no_brake else []
        again += ["--phones", "--phones-port", str(args.phones_port), "--phones-code", args.phones_code] if args.phones else []
        again += ["--landmarks", str(args.landmarks)] if args.landmarks or args.open_world else []
        again += ["--low-obstacles", str(args.low_obstacles)] if args.low_obstacles else []
        again += ["--sensors"] if args.sensors else []
        again += ["--tracks"] if args.tracks else []
        again += ["--zones"] if args.zones else []
        again += ["--replay"] + args.replay if args.replay else []
        again += ["--signs"] if args.signs else []
        again += ["--open-world", "--read", args.read] if args.open_world else []
        again += ["--voice-model", args.voice_model] if args.voice_model else []
        again += ["--walkers", args.walkers, "--crowd-scale", str(args.crowd_scale)] + (["--no-stream"] if args.no_stream else [])
        print(f"\nNext building: {name}", flush=True)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())] + again)


def run(world: Path, args):
    """Drive trips until --episodes, Q or Webots closing. Returns (building, seed, view) when N asks
    for the next building, which main() opens in a new Webots."""
    from baselines.drivers import make_driver
    from webots.bridge import Keyboard, WebotsScene

    scene = WebotsScene(world, walls=args.walls, animate=not args.fast, camera=args.camera,
                        trail=not (args.fast or args.no_trail or args.open_world), stream=not args.no_stream,
                        crowd_scale=args.crowd_scale, walkers=args.walkers)
    if args.model:
        from assist.driver import AssistDriver
        driver = AssistDriver(scene, args.model)
    else:
        driver = make_driver(args.driver, scene)
    views = None if args.fast else Views(scene, args.view)
    marks = landmark_module = None
    if args.landmarks:
        from webots import landmarks as landmark_module
        marks = landmark_module.Landmarks(scene.sup, scene.plan, world.stem, args.landmarks)
    lows = aim = sensors = trackview = zoneview = replay = guard = None
    if args.replay:
        from webots import replay as replay_module
        recorded = replay_module.load(args.replay[0])
        if not recorded:
            sys.exit(f"{args.replay[0]} has no tracks of {replay_module.MIN_TRACK_S:.0f} s or more (columns id,t,x,y).")
        replay = replay_module.Replay(scene, recorded, int(args.replay[1]) if len(args.replay) > 1 else 6)
        print(f"Replaying recorded tracks: {len(recorded)} in {args.replay[0]}, "
              f"median speed {np.median([t.speed() for t in recorded]):.2f} m/s", flush=True)
    if args.low_obstacles:
        from webots import obstacles
        lows = obstacles.LowObstacles(scene.sup, scene.plan, args.low_obstacles)
    if marks is not None and not args.fast:
        aim = landmark_module.Aim(scene.sup)
        for node in (aim.laser, aim.post, scene.sup.getFromDef("LANDMARK_SELECTOR")):
            scene.only_on_screen(node)          # the rider's pointer: not seen by the people walked from phones
    world_open = None
    if args.open_world:                        # hangs the signs itself
        from webots.open_world import OpenWorld
        world_open = OpenWorld(scene, driver, world.stem, read=args.read, target=args.target,
                               trail=not (args.fast or args.no_trail))   # its own trail, in pieces
    building_marks = None
    if world_open is not None and marks is not None:   # its markers: the whole building, up all trip
        from webots.open_world import BuildingMarkers
        building_marks = BuildingMarkers(scene, world.stem, [(f.x, f.y) for f in world_open.layout.faces])
        print(f"{len(building_marks.items)} markers up in the building", flush=True)
    elif args.signs:
        from webots import signs
        signs.for_world(scene.sup, scene.plan, world.stem)
    if not args.fast:
        from webots.sensorview import SensorView
        sensors = SensorView(scene)
        sensors.show(args.sensors)
        from webots.trackview import TrackView
        trackview = TrackView(scene)
        trackview.show(args.tracks)
        from webots.zoneview import ZoneView
        zoneview = ZoneView(scene)
        zoneview.show(args.zones)
        if args.zones and args.sensors:          # one see-through layer on the floor at a time
            sensors.show(False)
            print("The proxemic zones and the sensor fan flicker on top of each other, so the zones are "
                  "shown (S to show the fan instead)", flush=True)
        scene.listen_to_keyboard()             # so that quick taps on the keys are not missed
        from webots.guard import Guard
        guard = Guard(scene, brake=not args.no_brake)   # people in the way while the rider drives
    sensors_chosen = args.sensors and not args.zones     # S outside an override; while the rider drives it is on
    sensors_manual = False                        # whether the sensor view was last set for the rider driving
    key_help = KEY_HELP
    if world_open is not None:                 # the list keys, and with markers the marker keys while driving
        from webots import open_world
        key_help = open_world.KEY_HELP_MARKERS if marks is not None else open_world.KEY_HELP
    overlay = None if args.fast else Overlay(scene.sup, key_help)
    voice = None
    if not args.fast:                       # voice commands are always on (hold K), if the microphone and model are there
        from webots import voice as voice_module
        voice = voice_module.start(model=args.voice_model)
    players = phones = eyes = None
    if args.phones and not args.fast:
        from webots.phones import PhoneServer, WalkerEyes
        from webots.players import Players
        players = Players(scene, args.seed)
        phones = PhoneServer(args.phones_port, args.phones_code)
        eyes = WalkerEyes(scene)
        scene.before_moving = eyes.aim          # each phone's camera to its walker's eyes, every step
        args.phones_code = phones.code
        print(f"Phones on the same Wi-Fi: open {phones.address}  "
              f"(QR code: http://localhost:{args.phones_port}/{phones.code}/qr)", flush=True)
        if not os.environ.get("G19_NO_BROWSER"):
            import webbrowser
            webbrowser.open(f"http://localhost:{args.phones_port}/{phones.code}/qr")
    results, seed, last_trip, follow_up = [], args.seed, "No trips finished yet.", None
    print(f"{world.stem}: driver {args.model or args.driver}, walls from the {args.walls}. "
          f"Click the 3D view, then:\n    " + "\n    ".join(key_help), flush=True)
    quit_now = False
    while not quit_now and (not args.episodes or len(results) < args.episodes):
        if lows is not None:                    # before the trip is set up, so its start, goal and people avoid them
            lows.put_up(seed)
        obs, info = scene.reset(seed=seed)
        driver.reset()
        if trackview is not None:
            trackview.reset()
        if replay is not None:
            print(f"Recorded tracks replayed by {replay.begin(seed)} people", flush=True)
        scene.time_limit = float(scene.settings.get("time_limit_s", 60))
        if views is not None:                   # every new trip, restart or building: the camera
            views.show()                        # starts again on the chair, never where the last trip ended
        manual, paused, overrides, done, last_keys, gap = False, False, 0, False, set(), None
        going = None                              # the landmark the chair is driving to
        aim_point, aim_pose = None, None          # where the red line ends, and the pose it was last drawn from
        if marks is not None:                     # landmark mode: nothing to head for yet, and no trip clock
            scene.time_limit = math.inf
            if building_marks is None:
                new_landmarks(marks, scene, overlay, seed)
            else:                                 # the open world: its markers are up already
                marks.items = []
                list_markers(marks, building_marks, scene)
        listed_from = (scene.robot.x, scene.robot.y, scene.robot.orientation)   # where markers in sight were found
        if world_open is not None:                # open world: the chair knows nothing again, no trip clock
            world_open.begin()
        restart = False
        while not done:
            keys = scene.keys()
            pressed = keys - last_keys          # keys are reported while held: act once per press
            last_keys = keys
            if voice is not None:               # a spoken command comes in as the keys it stands for
                pressed |= voice.keys(keys, manual=manual, marks=marks, world_open=world_open)
            aiming = marks is not None and manual   # the number keys and G are for the markers
            if pressed & QUIT_KEYS:
                quit_now = True
                break
            if ord("R") in pressed:
                restart = True
                break
            if ord("N") in pressed:
                if args.attach:
                    print("N opens the next building in a new Webots, so it needs run.py without --attach.",
                          flush=True)
                else:
                    follow_up = (next_building(world), seed + (world_open is None), args.view)   # its camera starts afresh
                    quit_now = True
                    break
            if ord("L") in pressed and overlay is not None:
                overlay.show_keys(not overlay.keys_shown)
            if TEXT_KEY in pressed and overlay is not None:
                overlay.text_shown = not overlay.text_shown
                if overlay.last_legend is not None:   # the landmark list is only drawn when it changes
                    overlay.legend(*overlay.last_legend)
                print("Text at the top " + ("shown" if overlay.text_shown else "hidden (H shows it)"), flush=True)
            if phones is not None and not paused:      # phones walk people round the chair
                thumbs = phones.read()
                eyes.pitch = {slot: thumb.pitch for slot, thumb in thumbs.items()}
                for line in players.update(thumbs):
                    print(line, flush=True)
            if pressed & VIEW_KEYS and views is not None:
                views.toggle()
            if FIRST_PERSON_KEY in pressed and views is not None:
                views.toggle_first_person()
            if SENSOR_KEY in pressed and sensors is not None:
                sensors.show(not sensors.on)
                if not manual:                  # while the rider drives it is only for this time
                    sensors_chosen = sensors.on
                print("Sensor view " + ("on" if sensors.on else "off"), flush=True)
                if sensors.on and zoneview is not None and zoneview.on:     # one layer on the floor at a time
                    zoneview.show(False)
                    print("Proxemic zones off (they flicker under the sensor fan)", flush=True)
            if TRACKS_KEY in pressed and trackview is not None:
                trackview.show(not trackview.on)
                print("Tracks view " + ("on" if trackview.on else "off"), flush=True)
            if ZONES_KEY in pressed and zoneview is not None:
                zoneview.show(not zoneview.on)
                print("Proxemic zones " + ("on" if zoneview.on else "off"), flush=True)
                if zoneview.on and sensors is not None and sensors.on:      # one layer on the floor at a time
                    sensors.show(False)
                    if not manual:
                        sensors_chosen = False
                    print("Sensor view off (it flickers over the proxemic zones)", flush=True)
            if marks is not None and manual:         # the landmark selector works while the rider drives
                for digit in range(10):
                    if ord(str(digit)) in pressed:
                        picked = marks.selector.select(digit - 1 if digit else None)
                        print(f"Selected: {picked.name}" if picked else "Selected: none", flush=True)
                        if overlay is not None:
                            overlay.legend([lm.name for lm in marks.items], marks.selector.index)
            marker_go = voice is not None and voice.take_marker_go()     # a marker named while the chair drives
            if marks is not None and ord("G") in pressed and not paused and (aiming or world_open is None or marker_go):
                destination = marks.selector.selected
                if destination is None and manual and aim_point is not None:     # no marker: the red post
                    destination = landmark_module.Landmark("point ahead", "white", "point", *aim_point)
                if destination is None:
                    print("G drives to the marker locked on or picked: press O, then aim at one or press 1 to 8.", flush=True)
                else:   # the assistant takes over and drives to the spot in front of it
                    gx, gy = landmark_module.stopping_point(scene.plan, destination)
                    if world_open is not None:  # on the chair's own map, reading signs on the way
                        world_open.go_to_marker(destination.name, (gx, gy))
                    else:
                        scene.set_goal(gx, gy)  # moves the goal, plans the route, and fills it with people
                        getattr(getattr(driver, "controller", driver), "base", driver).path = None   # plan afresh
                    going, manual = destination, False
                    if aim is not None:
                        aim.hide()
                    print(f"Going to the {destination.name}: the assistant drives", flush=True)
                if marker_go:
                    pressed.discard(ord("G"))   # the marker's G, not the open world list's
            if world_open is not None and not paused and not aiming and world_open.keys(pressed):
                manual = False                  # G or X: the chair drives itself again
            if ord(" ") in pressed:
                paused = not paused
                print("PAUSED: Space resumes" if paused else "RESUMED", flush=True)
            if ord("O") in pressed and not paused:
                manual = not manual
                overrides += manual
                if manual and marks is not None and building_marks is None:   # new places to head for from here
                    new_landmarks(marks, scene, overlay, seed)
                if manual and marks is not None:
                    going, aim_pose = None, None
                if not manual and aim is not None:
                    aim.hide()
                print("MANUAL: arrows drive, O hands back" if manual else "AUTONOMOUS", flush=True)
            if sensors is not None and manual != sensors_manual:   # taking over: what the sensors see, on
                sensors.show((True if manual else sensors_chosen) and not (zoneview is not None and zoneview.on))
                sensors_manual = manual
            if guard is not None and not manual and guard.warning:
                guard.off()

            if paused:                          # everything holds still and the trip clock stops
                scene.hold_still()
            elif world_open is not None:        # the open world goes on until R, N or Q
                ride = manual_command(keys, Keyboard) if manual else None
                if ride is not None and guard is not None:
                    ride = guard.check(ride, info)
                obs, _, _, _, info = world_open.step(ride)
                r = scene.robot
                if building_marks is not None and (math.dist(listed_from[:2], (r.x, r.y)) > RELIST_M or abs(
                        (r.orientation - listed_from[2] + math.pi) % (2 * math.pi) - math.pi) > RELIST_RAD):
                    list_markers(marks, building_marks, scene)      # the markers in sight from here
                    listed_from = (r.x, r.y, r.orientation)
                if views is not None:
                    views.update()
                if sensors is not None:
                    sensors.update()
                if trackview is not None:
                    trackview.update(getattr(driver, "tracks", None) or [])
                if zoneview is not None:
                    zoneview.update()
                gap = info["MINIMUM_DISTANCE_TO_HUMAN"] - scene.ROBOT_RADIUS - scene.HUMAN_DIAMETER / 2
            else:
                command = driver.act(obs)       # always, so the assistant keeps tracking people
                if manual:
                    command = manual_command(keys, Keyboard)
                    if guard is not None:
                        command = guard.check(command, info)
                obs, _, terminated, truncated, info = scene.step(command)
                done = terminated or truncated
                if views is not None:
                    views.update()
                if sensors is not None:
                    sensors.update()
                if trackview is not None:
                    trackview.update(getattr(driver, "tracks", None) or [])
                if zoneview is not None:
                    zoneview.update()
                gap = info["MINIMUM_DISTANCE_TO_HUMAN"] - scene.ROBOT_RADIUS - scene.HUMAN_DIAMETER / 2
            if aim is not None and manual and not done and not paused:
                r = scene.robot                 # the red line along the chair's heading, as the rider looks
                end, length, locked = aim.update(scene.plan, r.x, r.y, r.orientation, marks.items)
                aim_point = (r.x + (end[0] - r.x) * max(0.0, 1 - 0.6 / max(length, 0.6)),
                             r.y + (end[1] - r.y) * max(0.0, 1 - 0.6 / max(length, 0.6)))
                pose = (round(r.x, 1), round(r.y, 1), round(r.orientation, 2))
                if pose != aim_pose:              # the chair moved or turned: the marker in line, if any, is picked
                    aim_pose = pose
                    if locked != marks.selector.index:
                        picked = marks.selector.select(locked)
                        if picked:
                            print(f"Locked on: {picked.name}", flush=True)
                if overlay is not None:
                    overlay.legend([lm.name for lm in marks.items], marks.selector.index,
                                   f"Aim: locked on the {marks.selector.selected.name} (G drives there)"
                                   if marks.selector.selected is not None else
                                   f"Aim: wall {length:.1f} m ahead (G drives to the red post)"
                                   if length < landmark_module.AIM_RANGE_M else
                                   f"Aim: nothing within {landmark_module.AIM_RANGE_M:.0f} m (G drives to the red post)")

            if overlay is not None:
                badge = ("PAUSED  (Space resumes)", GREY) if paused else \
                    (guard.warning, RED) if manual and guard is not None and guard.warning else \
                    ("MANUAL  (O hands back)" + (f"   people hit {guard.hits}" if guard is not None and guard.hits else "")
                     + (f"   walls hit {guard.wall_hits}" if guard is not None and guard.wall_hits else ""),
                     ORANGE) if manual else \
                    (world_open.badge(), GREEN) if world_open is not None else \
                    ("AUTONOMOUS  (O takes control)", GREEN)
                elapsed = scene.sup.getTime() - scene.start_time
                nearest = f"{gap:4.2f} m" if gap is not None and math.isfinite(gap) else "-"
                phones_line = (f"Phones: {phones.address}   {players.status()}" if phones is not None else "", GREY)
                if voice is not None:           # the voice line shares the last row with the phones
                    phones_line = ("   ".join(t for t in (phones_line[0], voice.line()) if t),
                                   ORANGE if voice.talking else GREY)
                if world_open is not None:      # what it is doing and knows, instead of the last trip
                    doing, knows = world_open.lines()
                    overlay.lines([
                        badge,
                        (doing, INK),
                        (f"{world.stem}   seed {seed}   {scene.robot.vel_x:3.1f} m/s   Nearest person {nearest}"
                         f"   Take-overs {overrides}", INK),
                        assistant_text(driver, manual),
                        (knows, GREY),
                        phones_line,
                    ])
                    if not (manual and marks is not None):   # while aiming, the markers are listed instead
                        world_open.legend(overlay)
                else:
                    overlay.lines([
                        badge,
                        (f"{world.stem}   seed {seed}   {elapsed:4.1f} s   {scene.robot.vel_x:3.1f} m/s", INK),
                        (f"Nearest person {nearest}   Take-overs {overrides}"
                         + (f"   Going to the {going.name}" if going is not None else ""), INK),
                        assistant_text(driver, manual),
                        (f"Last trip: {last_trip}" + (f"   {trackview.summary()}" if trackview is not None and trackview.on else ""), GREY),
                        phones_line,
                    ])
            if phones is not None:                  # what each walker sees, and how close the chair is
                r = scene.robot
                phones.set_frames(eyes.frames(), eyes.walkers())
                phones.publish(dict(trip=seed, time=scene.sup.getTime() - scene.start_time,
                                    chair=[r.x, r.y, r.orientation], status=assistant_text(driver, manual)[0],
                                    walkers=eyes.walkers()))
        if quit_now and not done:
            break
        if restart:
            print(f"seed {seed}: restarted", flush=True)
            seed += world_open is None                # the open world starts again just as it did
            continue
        outcome = "success" if info["SUCCESS"] else "hit a person" if info["COLLISION_HUMAN"] else \
            "hit a wall" if info["COLLISION_WALL"] else "timeout"
        print(f"seed {seed}: {outcome:12s} {info['TIME']:5.1f} s  {info['PATH_LENGTH']:5.1f} m  "
              f"overrides {overrides}", flush=True)
        last_trip = f"{outcome} in {info['TIME']:.1f} s, {info['PATH_LENGTH']:.1f} m"
        if trackview is not None and trackview.scores()[2]:
            print("   " + trackview.summary(), flush=True)
        results.append((outcome, info["TIME"], gap))
        seed += 1
    if phones is not None:
        phones.close()
    if voice is not None:
        voice.close()
    if results:
        n = len(results)
        ok = [t for o, t, _ in results if o == "success"]
        print(f"\n{n} trips: reached the goal {len(ok)} ({len(ok) / n:.0%}), "
              f"hit a person {sum(o == 'hit a person' for o, _, _ in results)}, "
              f"hit a wall {sum(o == 'hit a wall' for o, _, _ in results)}, "
              f"ran out of time {sum(o == 'timeout' for o, _, _ in results)}"
              + (f", average trip {np.mean(ok):.1f} s" if ok else ""))
    return follow_up


if __name__ == "__main__":
    main()
