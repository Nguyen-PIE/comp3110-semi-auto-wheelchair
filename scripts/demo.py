"""Watch the semi-autonomous wheelchair in the 2D simulator, and take control when you want to.

Examples:
    python scripts/demo.py                                  # S1 corridor
    python scripts/demo.py --scenario s3_concourse_high     # 20 people
    python scripts/demo.py --scenario fp_hospital           # real hospital floor plan
    python scripts/demo.py --scenario maps/level2.yaml      # your own floor plan

The chair always starts each episode driving itself, with the learned assistant
(models/assist_ppo.zip) in every scenario: it follows its route at full speed,
and slows down or steers round people only when someone is on its route or it
predicts someone will come too close. Press O to take control and steer with the
arrow keys; press O again to hand control back, and the chair carries on to the
same goal. Each take-over counts as an override.

A legend on the right of the window explains every symbol in the current
scene (press L to hide it).

Controls:
    O           take control / hand control back to the chair
    Arrow keys  steer while you have control (no keys held = stop)
    Space       pause / resume
    M           floor plans: switch between the view around the route and the whole map
    L           show / hide the legend
    R           restart the episode with a new seed
    N           next scenario
    Q or Esc    quit
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402  (imported before OpenCV, which avoids a clash between their SDL copies on macOS)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from baselines.drivers import DRIVERS, make_driver  # noqa: E402
from baselines.route_follower import plan_of  # noqa: E402
from envs.floorplan_env import ROUTE_BGR  # noqa: E402
from envs.make_env import SCENARIOS, VALIDATION_SEEDS, is_floorplan, is_map_path, make_env, scenario_title  # noqa: E402
from eval.metrics import EpisodeTracker  # noqa: E402
from legend import (DANGER_BGR, DESTINATION_BGR, LEGEND_W, PANEL_BG, PLAN_BGR,  # noqa: E402  (scripts/legend.py)
                    PREDICTION_BGR, build_legend, draw_dotted_line, draw_dotted_path, mark_standing_people)

MAX_VIEW_W, MAX_VIEW_H = 1000, 680
MIN_SCREEN_W = 820
HUD_H = 154
DRIVER_NAMES = {"AssistDriver": "learned assistant (RL)", "RuleDriver": "cautious rule (no learning)", "RouteFollower": "route follower",
                "GoToGoal": "go to goal", "RandomDriver": "random"}
MODE_STYLE = {  # badge text and colour for who is driving
    False: ("AUTONOMOUS: the chair is driving   (O to take control)", (30, 140, 60)),
    True: ("MANUAL: you are driving   (O to hand back)", (215, 110, 0)),
}
BG, INK, QUIET = (245, 245, 242), (25, 25, 25), (110, 110, 110)
DANGER_TEXT = (190, 30, 30)
OUTSIDE_BGR = (215, 215, 215)
VIEW_MARGIN_M = 3.0   # floor plans: space kept between the chair and the edge of the view
PREDICTION_DRAW_S = 2.0  # predicted paths are drawn this many seconds ahead
SHOW_PREDICTION_ARROWS = False  # arrows for where the assistant predicts people will be; hidden for now
OUTCOME_COLOURS = {"success": (30, 140, 60), "collision": (200, 40, 40), "timeout": (200, 130, 0)}


def fit_view(env) -> tuple[int, int]:
    """Choose the drawing size. Rooms are scaled to fit the window; floor plans are
    drawn at their own scale and the window shows the part with the start and goal."""
    u = env.unwrapped
    if is_floorplan(env):
        return min(MAX_VIEW_W, u.RESOLUTION_X), min(MAX_VIEW_H, u.RESOLUTION_Y)
    scale = min(MAX_VIEW_W / u.MAP_X, MAX_VIEW_H / u.MAP_Y)
    u.RESOLUTION_X = int(u.MAP_X * scale)
    u.RESOLUTION_Y = int(u.MAP_Y * scale)
    return u.RESOLUTION_X, u.RESOLUTION_Y


def to_pixel(u, x, y) -> tuple[int, int]:
    return int((x + u.MAP_X / 2) * u.PIXEL_TO_WORLD_X), int((u.MAP_Y / 2 - y) * u.PIXEL_TO_WORLD_Y)


def frame_box(u, start_xy, view_w, view_h) -> tuple[float, float, float, float]:
    """The part of the floor plan to show (x, y, width, height in plan pixels), at the plan's
    own scale. The view is centred on the chair's start, its goal and the route between them;
    when they do not all fit, it shifts just enough to keep the chair on screen."""
    points = [start_xy, (u.robot.goal_x, u.robot.goal_y)] + list(u.robot_route)
    pixels = np.array([to_pixel(u, x, y) for x, y in points], dtype=float)
    cx, cy = (pixels.min(axis=0) + pixels.max(axis=0)) / 2
    chair_x, chair_y = to_pixel(u, u.robot.x, u.robot.y)
    keep = min(VIEW_MARGIN_M * u.PIXEL_TO_WORLD_X, view_w / 2, view_h / 2)
    cx = float(np.clip(cx, chair_x - view_w / 2 + keep, chair_x + view_w / 2 - keep))
    cy = float(np.clip(cy, chair_y - view_h / 2 + keep, chair_y + view_h / 2 - keep))
    return cx - view_w / 2, cy - view_h / 2, view_w, view_h


def crop_to_view(img, box, view_w, view_h):
    """Cut `box` out of the full image (padding with grey outside it) and scale it to the window."""
    x0, y0, w, h = box
    ix0, iy0, iw, ih = int(round(x0)), int(round(y0)), max(1, int(round(w))), max(1, int(round(h)))
    region = np.empty((ih, iw, 3), dtype=np.uint8)
    region[:] = OUTSIDE_BGR
    H, W = img.shape[:2]
    sx0, sy0, sx1, sy1 = max(ix0, 0), max(iy0, 0), min(ix0 + iw, W), min(iy0 + ih, H)
    if sx1 > sx0 and sy1 > sy0:
        region[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = img[sy0:sy1, sx0:sx1]
    scale = view_w / iw
    if (iw, ih) != (view_w, view_h):
        region = cv2.resize(region, (view_w, view_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return region


def shortest_route(env) -> list[tuple[float, float]]:
    """The shortest route from the chair's start to its goal, around walls and furniture."""
    u = env.unwrapped
    if hasattr(u, "robot_route"):
        return list(u.robot_route)          # floor plans draw it themselves
    start, goal = (u.robot.x, u.robot.y), (u.robot.goal_x, u.robot.goal_y)
    return [start] + (plan_of(env).route(start, goal) or [goal])


def scene_image(env, driver, view_w, view_h, whole_map, start_xy, trail=None, guide=None) -> np.ndarray:
    """The picture for the window. Rooms are shown whole. Floor plans are shown at their own
    scale around the chair's start, goal and route (or the whole building with M). `trail` is
    the list of (x, y) positions the chair has driven through this episode; `guide` is the
    shortest route, drawn in green in the rooms (floor plans already show it)."""
    u = env.unwrapped
    # SocNavGym crosses out the picture when an episode ends; the demo shows a banner instead.
    ended = (u._is_terminated, u._is_truncated)
    u._is_terminated = u._is_truncated = False
    try:
        img = u.render_without_showing()  # BGR, shape (height, width, 3)
    finally:
        u._is_terminated, u._is_truncated = ended
    # Dotted line from each walking person to the ring marking where they are heading next.
    ppm = u.PIXEL_TO_WORLD_X
    person_r, ring_r = u.HUMAN_DIAMETER / 2, u.HUMAN_GOAL_RADIUS
    for human in u.dynamic_humans:
        if human.goal_x is None:
            continue
        dx, dy = human.goal_x - human.x, human.goal_y - human.y
        dist = math.hypot(dx, dy)
        if dist <= person_r + ring_r:
            continue  # already at the ring
        ux, uy = dx / dist, dy / dist
        start = to_pixel(u, human.x + ux * person_r, human.y + uy * person_r)
        end = to_pixel(u, human.goal_x - ux * ring_r, human.goal_y - uy * ring_r)
        draw_dotted_line(img, start, end, DESTINATION_BGR, spacing=max(5, int(0.2 * ppm)), radius=1 if ppm < 45 else 2)

    # The assistant's predictions: where each moving person will be in 2 s, red if they are
    # predicted to come within 0.3 m of the chair. Hidden unless SHOW_PREDICTION_ARROWS is True.
    for track in (getattr(driver, "tracks", None) or []) if SHOW_PREDICTION_ARROWS else []:
        if track.speed < 0.15:
            continue
        tip = to_pixel(u, track.x + track.vx * PREDICTION_DRAW_S, track.y + track.vy * PREDICTION_DRAW_S)
        colour = DANGER_BGR if track.collision_predicted else PREDICTION_BGR
        cv2.arrowedLine(img, to_pixel(u, track.x, track.y), tip, colour, 2, cv2.LINE_AA, tipLength=0.25)

    if guide and not is_floorplan(env):
        cv2.polylines(img, [np.array([to_pixel(u, x, y) for x, y in guide], dtype=np.int32)], False, ROUTE_BGR,
                      max(2, int(ppm * 0.12)), cv2.LINE_AA)
    # Where the chair has actually driven: a dotted orange line, to compare with the shortest
    # route (green) and the plan it is following now (solid orange).
    if trail and len(trail) > 1:
        draw_dotted_path(img, [to_pixel(u, x, y) for x, y in trail], PLAN_BGR,
                         spacing=max(6, int(0.25 * ppm)), radius=2 if ppm < 45 else 3)

    route = getattr(driver, "route", None)
    if route:
        points = [to_pixel(u, u.robot.x, u.robot.y)] + [to_pixel(u, x, y) for x, y in route]
        cv2.polylines(img, [np.array(points, dtype=np.int32)], False, PLAN_BGR, 2, cv2.LINE_AA)

    if is_floorplan(env):
        H, W = img.shape[:2]
        if whole_map:
            w = max(W, H * view_w / view_h)
            box = ((W - w) / 2, (H - w * view_h / view_w) / 2, w, w * view_h / view_w)
        else:
            box = frame_box(u, start_xy, view_w, view_h)
        img = crop_to_view(img, box, view_w, view_h)
    return img


def assistant_line(driver, manual: bool, small) -> list:
    """One status line saying what the assistant is doing (empty for drivers without one)."""
    if not hasattr(driver, "adjustment"):
        return []
    risky = sum(t.collision_predicted for t in driver.tracks)
    parts = []
    if manual:
        parts.append("Assistant: paused while you drive; resumes when you hand back")
    else:
        speed, offset = driver.adjustment
        degrees = math.degrees(offset)
        if not driver.controller.engaged and abs(degrees) < 2:
            parts.append("Assistant: route clear, following it")
        else:
            steer = "on the route" if abs(degrees) < 2 else f"{abs(degrees):.0f} deg {'left' if degrees > 0 else 'right'}"
            parts.append(f"Assistant: speed {speed:.0%}, steering {steer}")
        if driver.controller.safety_stop:
            parts.append("SAFETY STOP")
    if risky:
        parts.append(f"{risky} {'person' if risky == 1 else 'people'} predicted too close")
    return [("   |   ".join(parts), DANGER_TEXT if risky else INK, small)]


def manual_command() -> np.ndarray:
    """The user's command from the arrow keys: Up drives forward, Left/Right turn,
    nothing held stops the chair."""
    keys = pygame.key.get_pressed()
    forward = 1.0 if keys[pygame.K_UP] else 0.0
    turn = (1.0 if keys[pygame.K_LEFT] else 0.0) - (1.0 if keys[pygame.K_RIGHT] else 0.0)
    return np.array([forward, 0.0, turn], dtype=np.float32)


class Layout:
    """Where the scene, legend and text go in the window."""

    def __init__(self, view_w: int, view_h: int, legend: pygame.Surface | None):
        self.view_w, self.view_h, self.legend = view_w, view_h, legend
        legend_w = LEGEND_W if legend else 0
        self.top_h = max(view_h, legend.get_height() if legend else 0)
        self.size = (max(view_w + legend_w, MIN_SCREEN_W), self.top_h + HUD_H)
        self.scene_x = (self.size[0] - legend_w - view_w) // 2
        self.scene_y = (self.top_h - view_h) // 2


def draw(screen, font, img, layout: Layout, lines, manual: bool, banner=None):
    frame = pygame.surfarray.make_surface(np.ascontiguousarray(img[:, :, ::-1].swapaxes(0, 1)))
    screen.fill(BG)
    screen.blit(frame, (layout.scene_x, layout.scene_y))
    # Who is driving, in the top-left corner of the map.
    text, colour = MODE_STYLE[manual]
    label = font.render(text, True, (255, 255, 255))
    badge = pygame.Rect(layout.scene_x + 10, layout.scene_y + 10, label.get_width() + 20, label.get_height() + 10)
    pygame.draw.rect(screen, colour, badge, border_radius=6)
    screen.blit(label, (badge.x + 10, badge.y + 5))
    if layout.legend:
        x = screen.get_width() - LEGEND_W
        pygame.draw.rect(screen, PANEL_BG, (x, 0, LEGEND_W, layout.top_h))
        screen.blit(layout.legend, (x, 0))
    y = layout.top_h + 10
    for text, colour, f in lines:
        screen.blit(f.render(text, True, colour), (14, y))
        y += 26 if f is font else 22
    if banner:
        text, colour = banner
        label = font.render(text, True, (255, 255, 255))
        pad = 12
        box = pygame.Rect(0, 0, label.get_width() + 2 * pad, label.get_height() + 2 * pad)
        box.center = (layout.scene_x + layout.view_w // 2, layout.scene_y + layout.view_h // 2)
        pygame.draw.rect(screen, colour, box, border_radius=8)
        screen.blit(label, (box.x + pad, box.y + pad))
    pygame.display.flip()


def open_env(scenario: str):
    """make_env, but explains missing floor plans instead of crashing."""
    try:
        return make_env(scenario)
    except FileNotFoundError as exc:
        print(exc, flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="s1_corridor",
                        help=f"one of {', '.join(SCENARIOS)}, or the path to a floor-plan .yaml")
    parser.add_argument("--driver", default="auto",
                        help=f"the chair's automatic driver: {', '.join(DRIVERS)} "
                             "(auto = the learned assistant)")
    parser.add_argument("--seed", type=int, default=VALIDATION_SEEDS[0], help="first seed; each new episode adds 1")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    args = parser.parse_args()
    if args.scenario not in SCENARIOS and not is_map_path(args.scenario):
        parser.error(f"unknown scenario '{args.scenario}'")
    if args.driver == "keyboard":
        parser.error("--driver keyboard has been removed. Run the demo normally and press O to take control.")
    if args.driver not in DRIVERS:
        parser.error(f"unknown driver '{args.driver}'. Choose from: {', '.join(DRIVERS)}")

    scenario_names = list(SCENARIOS)
    scenario = args.scenario
    env = open_env(scenario)
    if env is None:
        sys.exit(1)

    pygame.init()
    pygame.display.set_caption("Wheelchair shared control: 2D simulator")
    font = pygame.font.SysFont("Helvetica,Arial", 20, bold=True)
    small = pygame.font.SysFont("Helvetica,Arial", 17)
    legend_title = pygame.font.SysFont("Helvetica,Arial", 17, bold=True)
    legend_text = pygame.font.SysFont("Helvetica,Arial", 15)
    clock = pygame.time.Clock()

    seed = args.seed
    last_result = "No episodes finished yet."
    whole_map = False
    show_legend = True
    fps = args.speed / env.unwrapped.TIMESTEP
    running = True
    while running:
        try:
            driver = make_driver(args.driver, env)
        except ValueError as exc:  # e.g. assist without a trained model
            print(f"{exc}. Using the automatic driver for this scenario instead.", flush=True)
            driver = make_driver("auto", env)
        obs, info = env.reset(seed=seed)
        driver.reset()
        mark_standing_people(env)
        view_w, view_h = fit_view(env)
        start_xy = (env.unwrapped.robot.x, env.unwrapped.robot.y)  # where the chair started
        trail = [start_xy]                                           # where it has driven since
        guide = shortest_route(env)                                  # the shortest route, for comparison
        show_plan = hasattr(driver, "route")
        show_predictions = SHOW_PREDICTION_ARROWS and hasattr(driver, "tracks")

        def make_layout() -> Layout:
            legend = build_legend(env, show_plan, legend_title, legend_text, show_predictions) if show_legend else None
            return Layout(view_w, view_h, legend)

        layout = make_layout()
        screen = pygame.display.set_mode(layout.size)
        tracker = EpisodeTracker(env)
        paused, done, restart = False, False, False
        manual = False  # every episode starts with the chair driving itself

        while running and not done and not restart:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_o:
                        manual = not manual
                        print(f"[{scenario} seed {seed}] {'You took control' if manual else 'Control handed back'} "
                              f"at {tracker.steps * env.unwrapped.TIMESTEP:.1f} s", flush=True)
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_m:
                        whole_map = not whole_map
                    elif event.key == pygame.K_l:
                        show_legend = not show_legend
                        layout = make_layout()
                        screen = pygame.display.set_mode(layout.size)
                    elif event.key == pygame.K_r:
                        restart = True
                    elif event.key == pygame.K_n:
                        start = scenario_names.index(scenario) if scenario in scenario_names else -1
                        for step in range(1, len(scenario_names) + 1):  # skip any whose map is missing
                            candidate = scenario_names[(start + step) % len(scenario_names)]
                            new_env = open_env(candidate)
                            if new_env is not None:
                                env.close()
                                env, scenario = new_env, candidate
                                break
                        restart = True
            if not running or restart:
                break

            if not paused:
                auto_action = driver.act(obs)  # always computed, so its plan is ready when you hand back
                action = manual_command() if manual else auto_action
                obs, reward, terminated, truncated, info = env.step(action)
                tracker.update(info, speed=float(env.unwrapped.robot.vel_x),
                               turn_rate=float(env.unwrapped.robot.vel_a), overridden=manual)
                done = terminated or truncated
                here = (env.unwrapped.robot.x, env.unwrapped.robot.y)
                if math.dist(here, trail[-1]) > 0.02:
                    trail.append(here)

            u = env.unwrapped
            gap = info.get("MINIMUM_DISTANCE_TO_HUMAN", math.inf) - u.ROBOT_RADIUS - u.HUMAN_DIAMETER / 2
            wall = f"   wall {info['DISTANCE_TO_WALL']:4.2f} m" if "DISTANCE_TO_WALL" in info else ""
            status = "PAUSED  " if paused else ""
            keys = "Space: pause   L: legend   R: new episode   N: next scenario   Q: quit"
            if is_floorplan(env):
                keys = keys.replace("L: legend", "M: whole map   L: legend")
            lines = [
                (f"{scenario_title(scenario)}   |   seed {seed}   |   automatic driver: {DRIVER_NAMES.get(type(driver).__name__, 'custom')}", INK, font),
                (f"{status}time {tracker.steps * u.TIMESTEP:4.1f} s   speed {u.robot.vel_x:3.1f} m/s   "
                 f"nearest person {gap:4.2f} m{wall}   near misses {tracker.near_misses}   "
                 f"overrides {tracker.overrides} ({tracker.manual_steps * u.TIMESTEP:.1f} s manual)", INK, small),
                *assistant_line(driver, manual, small),
                (f"Last episode: {last_result}", QUIET, small),
                ("O: take control / hand back   Arrows: steer while you have control (no keys = stop)", QUIET, small),
                (keys, QUIET, small),
            ]
            draw(screen, font, scene_image(env, driver, view_w, view_h, whole_map, start_xy, trail, guide), layout, lines, manual)
            clock.tick(fps)

        if done:
            s = tracker.summary()
            what = s["outcome"].upper()
            if s["collision"]:
                what += " (wall)" if s["collision_wall"] else " (person)" if s["collision_human"] else ""
            last_result = (f"{what} in {s['time_s']} s, path {s['path_length_m']} m, "
                           f"closest {s['min_gap_to_person_m']} m, near misses {s['near_misses']}, "
                           f"overrides {s['overrides']} ({s['manual_time_s']} s manual)")
            print(f"[{scenario} seed {seed}] {last_result}", flush=True)
            draw(screen, font, scene_image(env, driver, view_w, view_h, whole_map, start_xy, trail, guide), layout, lines, manual,
                 banner=(what, OUTCOME_COLOURS[s["outcome"]]))
            pygame.time.wait(1500)
        seed += 1

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
