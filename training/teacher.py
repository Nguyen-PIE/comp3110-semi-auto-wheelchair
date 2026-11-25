"""A hand-written, cautious version of the assistant, used to give the policy a head start.

Reinforcement learning from scratch spends a long time crashing before it learns to
brake. So training/train_assist.py first copies this rule (behaviour cloning), then
improves on it with PPO. The rule reads the same people tracker as the policy:

  - someone ahead of the chair and close: stop (closer than 0.5 m straight
    ahead, less for people off to the side, 0.1 m for someone brushing past);
  - someone predicted to come within 0.3 m in the next 3 s: slow down, more the
    sooner it will happen, and steer away from where they will be;
  - someone standing in the chair's path ahead: steer around them.

It picks a side to pass each person on once and keeps to it, as people do. When
it is not clear which side is better (someone walking straight at the chair), it
takes the side with more room from the walls, and keeps left if both have room.
Choosing again every step made the chair swerve left and right in front of
people walking towards it.
"""

from __future__ import annotations

import math

import numpy as np

from assist.sensors import N_RAYS

STOP_GAP_M = 0.5        # stop for anyone in the chair's path closer than this
LOOK_AHEAD_S = 3.0      # react to predicted collisions this far ahead
AROUND_M = 2.5          # steer around people standing in the path this far ahead
UNCLEAR_M = 0.3         # a predicted sideways offset smaller than this does not decide the side
WRONG_SIDE_M = 0.4      # change sides only if the person will clearly be on the chosen side
FORGET_STEPS = 10       # forget the side chosen for someone after 1 s without needing it
MORE_ROOM_M = 0.5       # one side has more room if its wall gap is this much bigger


def _room(controller, side: int) -> float:
    """Distance to the wall on the chair's left (side 1) or right (side -1)."""
    quarter = N_RAYS // 4
    rays = [quarter - 1, quarter, quarter + 1] if side > 0 else [3 * quarter - 1, 3 * quarter, 3 * quarter + 1]
    return float(min(controller.ranges[i] for i in rays))


def _side(controller, track_id: int, offset: float) -> int:
    """Which way to steer around a person (1 left, -1 right). `offset` is where they will be,
    sideways from the chair (+ left). Kept from step to step for the same person."""
    chosen = controller.sides.get(track_id)
    if chosen is not None and controller.steps - chosen[1] <= FORGET_STEPS:
        side = chosen[0]
        if side * offset > WRONG_SIDE_M:      # they will be clearly on the side we were steering to
            side = -side
    elif abs(offset) >= UNCLEAR_M:
        side = -1 if offset > 0 else 1        # steer away from where they will be
    else:
        left, right = _room(controller, 1), _room(controller, -1)
        side = 1 if left > right - MORE_ROOM_M else -1   # more room, or keep left
    controller.sides[track_id] = (side, controller.steps)
    return side


def teacher_action(controller, robot) -> np.ndarray:
    """The rule's action, in the policy's format: [speed (-1 stop to 1 full), steering (-1 right to 1 left)]."""
    c, s = math.cos(robot.orientation), math.sin(robot.orientation)
    cvx, cvy = robot.vel_x * c, robot.vel_x * s
    speed, steer = 1.0, 0.0
    soonest = math.inf
    for t in controller.tracks:
        px, py = t.x - robot.x, t.y - robot.y
        ahead, left = c * px + s * py, -s * px + c * py           # position in the chair's frame
        towards = ahead / max(math.hypot(px, py), 1e-6)            # 1 straight ahead, 0 beside, below 0 behind
        in_lane = ahead > 0 and abs(left) < robot.radius + t.radius + 0.15   # in the strip the chair drives through
        if (towards > 0 and t.distance < STOP_GAP_M * towards) or (towards > -0.5 and t.distance < 0.1):
            # less room needed for people off to the side; wait for anyone brushing past its side
            speed = 0.0
            if soonest > 0:
                soonest, steer = 0.0, float(_side(controller, t.id, left))
        if t.collision_predicted and t.t_closest < LOOK_AHEAD_S:
            speed = min(speed, float(np.clip((t.t_closest - 0.5) / 2.5, 0.0, 1.0)))
            if t.t_closest < soonest:
                fx = px + (t.vx - cvx) * t.t_closest
                fy = py + (t.vy - cvy) * t.t_closest
                left_then = -s * fx + c * fy                     # where they will be, sideways
                soonest = t.t_closest
                steer = _side(controller, t.id, left_then) * float(np.clip(1.5 - t.t_closest / 2, 0.3, 1.0))
        elif t.id in controller.still and in_lane and ahead < AROUND_M and soonest == math.inf:
            steer = float(_side(controller, t.id, left))
    return np.array([speed * 2.0 - 1.0, steer], dtype=np.float32)
