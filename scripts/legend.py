"""The legend panel shown beside the simulator in scripts/demo.py.

Each symbol is drawn with the simulator's own drawing code (or the same colours),
so the legend always matches the scene. Only things present in the current
scenario are listed.
"""

from __future__ import annotations

import copy
import math

import cv2
import numpy as np
import pygame

from envs.floorplan_env import ROUTE_BGR, UNKNOWN_BGR, WALL_BGR

LEGEND_W = 260
PAD = 14
SWATCH_W, SWATCH_H = 52, 34
ROW_GAP = 8
PANEL_BG = (236, 236, 231)
INK, QUIET = (25, 25, 25), (95, 95, 95)

# Colours used in the scene (BGR, as OpenCV draws them).
STANDING_BGR = (210, 80, 180)   # purple: people standing still (the demo recolours them)
PLAN_BGR = (0, 140, 255)        # orange: the automatic driver's current plan (solid) and the path driven (dotted)
DESTINATION_BGR = (120, 0, 0)   # navy: SocNavGym's marker for where a person is heading
PREDICTION_BGR = (110, 110, 110)  # grey: a person's predicted path (assistant)
DANGER_BGR = (40, 40, 220)        # red: predicted to come too close to the chair
ROOM_WALL_BGR = (0, 0, 0)       # SocNavGym draws room walls black
GROUP_LINE_BGR = (0, 0, 255)    # SocNavGym joins people in a group with a red line


def mark_standing_people(env) -> None:
    """Draw people who stand still (alone or in a group) in purple, so they can
    be told apart from people walking (blue). Display only; nothing else changes."""
    u = env.unwrapped
    for person in u.static_humans + [h for group in u.static_interactions for h in group.humans]:
        person.color = STANDING_BGR


def _canvas() -> np.ndarray:
    return np.full((SWATCH_H, SWATCH_W, 3), 255, dtype=np.uint8)


def _centre() -> tuple[int, int]:
    return SWATCH_W // 2, SWATCH_H // 2


def _object(obj, size_m: float, **attrs) -> np.ndarray:
    """Draw a copy of a simulator object in the middle of a swatch, scaled to fit."""
    img = _canvas()
    ppm = min(45.0, (SWATCH_H - 4) / max(size_m, 1e-3), (SWATCH_W - 4) / max(size_m, 1e-3))
    o = copy.deepcopy(obj)
    o.x, o.y = 0.0, 0.0
    for key, value in attrs.items():
        setattr(o, key, value)
    o.draw(img, ppm, ppm, SWATCH_W / ppm, SWATCH_H / ppm)
    return img


def _group(person) -> np.ndarray:
    """Two people facing each other, joined by the group line."""
    img = _canvas()
    ppm = 36.0
    centres = []
    for x, facing in ((-0.45, 0.0), (0.45, math.pi)):
        o = copy.deepcopy(person)
        o.x, o.y, o.orientation, o.color = x, 0.0, facing, STANDING_BGR
        o.draw(img, ppm, ppm, SWATCH_W / ppm, SWATCH_H / ppm)
        centres.append((int(SWATCH_W / 2 + x * ppm), SWATCH_H // 2))
    cv2.line(img, centres[0], centres[1], GROUP_LINE_BGR, 1)
    return img


def _goal() -> np.ndarray:
    img = _canvas()
    cv2.circle(img, _centre(), 14, (0, 180, 0), -1)
    cv2.circle(img, _centre(), 14, (0, 255, 0), 2)
    return img


def draw_dotted_line(img, start, end, colour, spacing: int = 7, radius: int = 1) -> None:
    """A dotted line from start to end (pixel coordinates). OpenCV has no dotted line style."""
    (x0, y0), (x1, y1) = start, end
    length = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(length // spacing))
    for i in range(steps + 1):
        t = i / steps
        cv2.circle(img, (int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))), radius, colour, -1)


def draw_dotted_path(img, points, colour, spacing: int = 8, radius: int = 2) -> None:
    """Dots every `spacing` pixels along a polyline (pixel coordinates), evenly spaced across its
    corners, so a path drawn step by step looks like one dotted line."""
    carry = 0.0   # distance along the path to the next dot
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        d = carry
        while d <= length:
            t = d / length if length else 0.0
            cv2.circle(img, (int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))), radius, colour, -1,
                       cv2.LINE_AA)
            d += spacing
        carry = d - length


def _dotted(colour) -> np.ndarray:
    """A dotted line, as drawn for the path the chair has driven."""
    img = _canvas()
    draw_dotted_path(img, [(5, SWATCH_H - 8), (SWATCH_W - 6, 8)], colour, spacing=6, radius=2)
    return img


def _destination(colour) -> np.ndarray:
    """A ring with a dotted line leading to it, as drawn for each walking person."""
    img = _canvas()
    centre = (SWATCH_W - 14, SWATCH_H // 2)
    cv2.circle(img, centre, 9, colour, 2)
    draw_dotted_line(img, (4, SWATCH_H // 2), (centre[0] - 12, SWATCH_H // 2), colour, spacing=5)
    return img


def _block(colour, border=None) -> np.ndarray:
    img = _canvas()
    cv2.rectangle(img, (6, 5), (SWATCH_W - 7, SWATCH_H - 6), colour, -1)
    if border:
        cv2.rectangle(img, (6, 5), (SWATCH_W - 7, SWATCH_H - 6), border, 1)
    return img


def _bar(colour) -> np.ndarray:
    img = _canvas()
    cv2.rectangle(img, (4, SWATCH_H // 2 - 3), (SWATCH_W - 5, SWATCH_H // 2 + 3), colour, -1)
    return img


def _line(colour, thickness) -> np.ndarray:
    img = _canvas()
    cv2.line(img, (5, SWATCH_H - 8), (SWATCH_W - 6, 8), colour, thickness, cv2.LINE_AA)
    return img


def _arrow(colour) -> np.ndarray:
    img = _canvas()
    cv2.arrowedLine(img, (6, SWATCH_H // 2 + 6), (SWATCH_W - 6, SWATCH_H // 2 - 6), colour, 2, cv2.LINE_AA, tipLength=0.3)
    return img


def legend_entries(env, show_driver_plan: bool, show_predictions: bool = False) -> list[tuple[np.ndarray, str]]:
    """(swatch image, label) for everything in the current scene."""
    u = env.unwrapped
    floorplan = hasattr(u, "plan")
    entries = [
        (_object(u.robot, 2 * u.robot.radius + 0.05, orientation=0.0), "Wheelchair (arrow = direction it faces)"),
        (_goal(), "Goal the chair must reach"),
    ]
    walkers = u.dynamic_humans + [h for g in u.moving_interactions for h in g.humans]
    standers = u.static_humans + [h for g in u.static_interactions for h in g.humans]
    any_person = (walkers + standers + [None])[0]
    if walkers:
        entries.append((_object(walkers[0], 0.78, orientation=math.pi / 2, color=None),
                        "Person walking, seen from above"))
        entries.append((_destination(DESTINATION_BGR), "Where that person is heading next (dotted line from the person)"))
    if u.static_humans:
        entries.append((_object(u.static_humans[0], 0.78, orientation=math.pi / 2, color=STANDING_BGR),
                        "Person standing still"))
    if (u.static_interactions or u.moving_interactions) and any_person is not None:
        entries.append((_group(any_person), "Group talking (joined by a red line)"))
    for things, label in ((u.tables, "Table"), (u.chairs, "Chair"), (u.plants, "Potted plant"),
                          (u.laptops, "Laptop on a table")):
        if things:
            t = things[0]
            size = max(getattr(t, "length", 0) or 0, getattr(t, "width", 0) or 0, 2 * (getattr(t, "radius", 0) or 0))
            entries.append((_object(t, size, orientation=0.0), label))
    if floorplan:
        entries.append((_block(WALL_BGR), "Wall or fixed furniture (touching = collision)"))
        entries.append((_block(UNKNOWN_BGR, border=(170, 170, 170)), "Closed room, or outside the building"))
        entries.append((_line(ROUTE_BGR, 5), "Shortest route, start to goal (guide only)"))
    else:
        entries.append((_bar(ROOM_WALL_BGR), "Wall (touching = collision)"))
        entries.append((_line(ROUTE_BGR, 5), "Shortest route, start to goal (guide only)"))
    if show_driver_plan:
        entries.append((_line(PLAN_BGR, 2), "Automatic driver's current plan"))
    entries.append((_dotted(PLAN_BGR), "Path the chair has actually driven (dotted)"))
    if show_predictions:
        entries.append((_arrow(PREDICTION_BGR), "Where the chair predicts a person will be in 2 s"))
        entries.append((_arrow(DANGER_BGR), "Same, red when they are predicted to come within 0.3 m"))
    return entries


def numbers_help(floorplan: bool) -> list[tuple[str, str]]:
    rows = [
        ("nearest person", "edge-to-edge gap to the closest person"),
        ("near miss", "passing within 0.3 m of a person"),
        ("overrides", "times you pressed O to take control, and how long you drove"),
    ]
    if floorplan:
        rows.append(("wall", "edge-to-edge gap to the nearest wall"))
    return rows


def _wrap(text: str, font, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if font.size(trial)[0] <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build_legend(env, show_driver_plan: bool, title_font, font, show_predictions: bool = False) -> pygame.Surface:
    """The legend panel as a pygame surface, LEGEND_W wide and as tall as it needs."""
    text_x = PAD + SWATCH_W + 10
    text_w = LEGEND_W - text_x - PAD
    line_h = font.get_linesize()
    blocks = []  # (y, surface or callable)
    y = PAD
    title = title_font.render("Legend", True, INK)
    blocks.append((PAD, y, title))
    y += title.get_height() + 8

    for img, label in legend_entries(env, show_driver_plan, show_predictions):
        swatch = pygame.surfarray.make_surface(np.ascontiguousarray(img[:, :, ::-1].swapaxes(0, 1)))
        lines = _wrap(label, font, text_w)
        row_h = max(SWATCH_H, len(lines) * line_h)
        blocks.append((PAD, y + (row_h - SWATCH_H) // 2, swatch))
        ty = y + (row_h - len(lines) * line_h) // 2
        for line in lines:
            blocks.append((text_x, ty, font.render(line, True, INK)))
            ty += line_h
        y += row_h + ROW_GAP

    y += 6
    heading = title_font.render("Numbers below the map", True, INK)
    blocks.append((PAD, y, heading))
    y += heading.get_height() + 6
    for term, meaning in numbers_help(hasattr(env.unwrapped, "plan")):
        for line in _wrap(f"{term}: {meaning}", font, LEGEND_W - 2 * PAD):
            blocks.append((PAD, y, font.render(line, True, QUIET)))
            y += line_h
        y += 4

    panel = pygame.Surface((LEGEND_W, y + PAD))
    panel.fill(PANEL_BG)
    pygame.draw.line(panel, (205, 205, 200), (0, 0), (0, panel.get_height()), 1)
    for x, by, surface in blocks:
        panel.blit(surface, (x, by))
    return panel
