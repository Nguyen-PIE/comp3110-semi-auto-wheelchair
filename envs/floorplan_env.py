"""SocNavGym with a real building floor plan instead of an empty rectangular room.

What changes compared with SocNavGym-v2:
  - Walls come from the floor plan (envs/floorplan.py), checked against a
    distance map rather than thousands of wall objects, so it stays fast.
  - The chair starts and finishes at reachable points 10 to 25 m apart along
    the route (configurable), never inside a wall or a closed room.
  - People start and choose destinations within a few metres of the chair's
    route, so encounters actually happen in a large building. They walk along
    routes around the walls and around the chair (A* on the floor plan) and
    are pushed back if the crowd squeezes them into a wall. Anyone whose
    destination the chair is sitting on picks another one.
  - SocNavGym's reference "ORCA robot" is switched off, because it drives
    straight through walls.

Limits for now: the observation does not include the floor-plan walls yet
(obs["walls"] is all zeros), and moving groups are not supported (standing
groups are). Everything else (people, rewards, info) is SocNavGym's.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from envs import socnavgym_fixes  # noqa: F401  (patches SocNavGym bugs; see that file)
from socnavgym.envs.socnavenv_v2 import SocNavEnv_v2, SocNavGymObject
from socnavgym.envs.utils.plant import Plant
from socnavgym.envs.utils.utils import w2px, w2py

from envs.floorplan import FREE, UNKNOWN, WALL, FloorPlan, load_floorplan, route_length

PASSING_ROOM_M = 1.0       # standing people and groups leave at least this gap to the walls

WALL_BGR = (70, 70, 70)
UNKNOWN_BGR = (215, 215, 215)
FREE_BGR = (255, 255, 255)
ROUTE_BGR = (170, 225, 170)


class FloorPlanSocNavEnv(SocNavEnv_v2):
    def __init__(self, config: str | None = None, render_mode: str | None = None, *,
                 floorplan: str | Path | FloorPlan,
                 min_goal_distance: float = 10.0, max_goal_distance: float = 25.0,
                 people_within: float = 6.0, pixels_per_metre: float = 30.0):
        self.plan = floorplan if isinstance(floorplan, FloorPlan) else load_floorplan(str(floorplan))
        self.min_goal_distance = min_goal_distance
        self.max_goal_distance = max_goal_distance
        self.people_within = people_within
        self.robot_route: list[tuple[float, float]] = []
        self._start = self._goal = (0.0, 0.0)
        self._focus_cells = self.plan.spawn_cells
        self._route_version = 0
        self._plan_image = self._backdrop_image = self._backdrop_key = None
        super().__init__(config=config, render_mode=render_mode)

        # The map is the floor plan, so its size never changes.
        self.set_shape = "rectangle"
        self.MIN_MAP_X = self.MAX_MAP_X = self.MAP_X = self.plan.width_m
        self.MIN_MAP_Y = self.MAX_MAP_Y = self.MAP_Y = self.plan.height_m
        self.RESOLUTION_X = int(round(self.plan.width_m * pixels_per_metre))
        self.RESOLUTION_Y = int(round(self.plan.height_m * pixels_per_metre))

    # ------------------------------------------------------------------ reset
    def randomize_params(self):
        super().randomize_params()
        self.MAP_X, self.MAP_Y = self.plan.width_m, self.plan.height_m
        self._choose_start_and_goal()

    def _choose_start_and_goal(self):
        """Pick the chair's start and goal so the route between them is within range."""
        plan = self.plan
        best = None        # (how far outside the range, start, goal, route)
        farthest = None    # fallback for small plans: the pair furthest apart
        for _ in range(300):
            start, goal = plan.random_point(random), plan.random_point(random)
            straight = math.dist(start, goal)
            if straight > self.max_goal_distance:
                continue
            if straight < self.min_goal_distance:
                if farthest is None or straight > farthest[0]:
                    farthest = (straight, start, goal)
                continue
            route = plan.route(start, goal)
            if route is None:
                continue
            excess = max(0.0, route_length(start, route) - self.max_goal_distance)
            if best is None or excess < best[0]:
                best = (excess, start, goal, route)
            if excess == 0.0:
                break
        if best is None and farthest is not None:
            _, start, goal = farthest
            route = plan.route(start, goal) or [goal]
            best = (0.0, start, goal, route)
        if best is None:
            raise RuntimeError(f"Could not find a start and goal in floor plan '{plan.name}'")
        _, self._start, self._goal, route = best
        self.robot_route = [self._start] + list(route)
        self._route_version += 1
        self._focus_cells = plan.cells_near(self.robot_route, self.people_within)

    def _room_needed(self, object_type, kwargs) -> float:
        """How far from walls an object's centre must be."""
        human_r = self.HUMAN_DIAMETER / 2
        group_r = self.INTERACTION_RADIUS + human_r
        if object_type == SocNavGymObject.DYNAMIC_HUMAN:
            return human_r + 0.2
        if object_type == SocNavGymObject.STATIC_HUMAN:
            return human_r + PASSING_ROOM_M
        if object_type == SocNavGymObject.PLANT:
            return kwargs["radius"] + PASSING_ROOM_M
        if object_type in (SocNavGymObject.TABLE, SocNavGymObject.CHAIR):
            return math.hypot(kwargs["width"], kwargs["length"]) / 2 + PASSING_ROOM_M
        if object_type in (SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC,
                           SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC_NON_DISPERSING):
            return group_r + 0.2
        return group_r + PASSING_ROOM_M  # standing groups

    def _get_kwargs(self, object_type, extra_info=None):
        kwargs = super()._get_kwargs(object_type, extra_info)
        if object_type == SocNavGymObject.ROBOT:
            kwargs["x"], kwargs["y"] = self._start
            return kwargs
        if "x" not in kwargs or object_type == SocNavGymObject.LAPTOP:
            return kwargs  # laptops sit on tables; human-laptop pairs sit at laptops
        need = self._room_needed(object_type, kwargs)
        keep_away = 1.0 if object_type == SocNavGymObject.DYNAMIC_HUMAN else 1.5
        x, y = self._start
        for attempt in range(120):
            cells = self._focus_cells if attempt < 60 else None  # widen the search if the route area is full
            x, y = self.plan.random_point(random, cells)
            if self.plan.clearance_at(x, y) < need:
                continue
            if math.dist((x, y), self._start) < keep_away or math.dist((x, y), self._goal) < keep_away:
                continue
            break
        kwargs["x"], kwargs["y"] = x, y
        return kwargs

    def sample_goal(self, goal_radius, HALF_SIZE_X, HALF_SIZE_Y):
        """A destination someone can actually walk to, near the chair's route, and not where
        the chair is now."""
        others = [o for o in self.objects + list(self.goals.values()) if o is not None]
        for attempt in range(200):
            cells = self._focus_cells if attempt < 100 else None
            x, y = self.plan.random_point(random, cells)
            goal = Plant(id=None, x=x, y=y, radius=goal_radius)
            if not any(goal.collides(o) for o in others) and not self._chair_on(x, y):
                return goal
        return None

    def _chair_on(self, x, y) -> bool:
        """Whether the chair takes up the spot (x, y), so a person cannot stand there."""
        robot = getattr(self, "robot", None)
        if robot is None or robot.x is None:
            return False
        return math.dist((x, y), (robot.x, robot.y)) < robot.radius + self.HUMAN_DIAMETER / 2 + 0.4

    def _add_walls(self):
        pass  # the floor plan provides the walls

    def try_reset(self, seed=None, options=None):
        ok, obs, info = super().try_reset(seed, options)
        if not ok:
            return ok, obs, info
        # SocNavGym drew its own goal for the chair; use the one with a good route instead.
        gx, gy = self._goal
        self.goals[self.robot.id] = Plant(id=None, x=gx, y=gy, radius=self.GOAL_RADIUS)
        self.robot.goal_x, self.robot.goal_y = gx, gy
        self.robot_orca.goal_x, self.robot_orca.goal_y = gx, gy
        self.has_orca_robot_collided = True  # switch off SocNavGym's wall-blind reference robot
        self.prev_goal_distance = math.dist((self.robot.x, self.robot.y), (gx, gy))
        self._prev_observations, self._current_observations = {}, {}
        self.populate_prev_obs()
        obs = self._get_obs()
        self.reward_calculator.re_init(self)
        return True, obs, info

    # ------------------------------------------------------------------- step
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        clearance = self.plan.clearance_at(self.robot.x, self.robot.y)
        info["DISTANCE_TO_WALL"] = clearance - self.robot.radius  # gap between the chair's edge and the nearest wall
        info["MINIMUM_OBSTACLE_DISTANCE"] = min(info.get("MINIMUM_OBSTACLE_DISTANCE", math.inf), clearance)
        return obs, reward, terminated, truncated, info

    def handle_collision_and_update(self):
        before = socnavgym_fixes.people_positions(self)
        super().handle_collision_and_update()
        radius = self.HUMAN_DIAMETER / 2
        for human in self.dynamic_humans + [h for group in self.moving_interactions for h in group.humans]:
            human.x, human.y = self.plan.push_out(human.x, human.y, radius)
        socnavgym_fixes.step_back_from_chair(self, before)  # being pushed off a wall must not push them into the chair

    def check_robot_collision(self, robot):
        collision_human, collision_object, collision_wall = super().check_robot_collision(robot)
        if self.plan.clearance_at(robot.x, robot.y) < robot.radius:
            collision_object = collision_wall = True
        return collision_human, collision_object, collision_wall

    def compute_orca_interaction_velocities(self):
        if not self.moving_interactions:
            return []  # nothing to compute; saves building an ORCA simulation every step
        return super().compute_orca_interaction_velocities()

    # ---------------------------------------------------------------- drawing
    def _backdrop(self) -> np.ndarray:
        """The floor plan at the current drawing size, with the chair's route on it."""
        size = (int(self.RESOLUTION_X), int(self.RESOLUTION_Y))
        if self._plan_image is None or self._plan_image.shape[1::-1] != size:
            palette = np.zeros((3, 3), dtype=np.uint8)
            palette[FREE], palette[WALL], palette[UNKNOWN] = FREE_BGR, WALL_BGR, UNKNOWN_BGR
            image = palette[self.plan.cells]
            shrinking = size[0] < image.shape[1]
            self._plan_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_NEAREST)
            self._backdrop_key = None
        key = (size, self._route_version)
        if self._backdrop_key != key:
            image = self._plan_image.copy()
            if len(self.robot_route) > 1:
                pts = np.array([[(x + self.MAP_X / 2) * self.PIXEL_TO_WORLD_X, (self.MAP_Y / 2 - y) * self.PIXEL_TO_WORLD_Y]
                                for x, y in self.robot_route], dtype=np.int32)
                cv2.polylines(image, [pts], False, ROUTE_BGR, max(2, int(self.PIXEL_TO_WORLD_X * 0.12)), cv2.LINE_AA)
            self._backdrop_image, self._backdrop_key = image, key
        return self._backdrop_image

    def render_without_showing(self, mode="human", draw_human_gaze=False, draw_human_goal=True):
        """Draw the scene on the floor plan (BGR image, one pixel = 1 / pixels_per_metre m).

        Same picture as SocNavGym's version, but starts from the cached floor
        plan instead of a blank page, which is much faster for big buildings.
        """
        px, py, mx, my = self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y
        img = self._backdrop().copy()
        self.world_image = img
        for thing in self.tables + self.chairs + self.laptops + self.plants:
            thing.draw(img, px, py, mx, my)

        goal = (w2px(self.robot.goal_x, px, mx), w2py(self.robot.goal_y, py, my))
        radius = max(2, int(self.GOAL_RADIUS * px))
        cv2.circle(img, goal, radius, (0, 180, 0), -1)
        cv2.circle(img, goal, radius, (0, 255, 0), 2)

        if draw_human_goal:  # where each walking person is heading next
            for human in self.dynamic_humans:
                cv2.circle(img, (w2px(human.goal_x, px, mx), w2py(human.goal_y, py, my)),
                           max(2, int(self.HUMAN_GOAL_RADIUS * px)), (120, 0, 0), 2)
        for human in self.static_humans + self.dynamic_humans:
            if draw_human_gaze:
                human.draw_gaze_range(img, self.HUMAN_GAZE_ANGLE, px, py, mx, my)
            human.draw(img, px, py, mx, my)
        self.robot.draw(img, px, py, mx, my)
        for group in self.moving_interactions + self.static_interactions + self.h_l_interactions:
            group.draw(img, px, py, mx, my)
        return img


if "WheelchairFloorPlan-v0" not in gym.registry:
    gym.register(id="WheelchairFloorPlan-v0", entry_point=FloorPlanSocNavEnv)
