"""Check that everything needed for the 2D simulator is installed and working.

Run:  python scripts/check_install.py
It needs no window, so it also works over SSH.
"""

from __future__ import annotations

import importlib
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(label: str, fn) -> bool:
    try:
        detail = fn()
        print(f"  OK    {label}{f'  ({detail})' if detail else ''}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
        return False


def version(module: str):
    return lambda: getattr(importlib.import_module(module), "__version__", "installed")


def rvo2_works():
    import rvo2
    sim = rvo2.PyRVOSimulator(0.1, 1.5, 5, 1.5, 2, 0.4, 2)
    agent = sim.addAgent((0, 0))
    sim.setAgentPrefVelocity(agent, (1, 0))
    sim.doStep()
    x, _ = sim.getAgentPosition(agent)
    assert x > 0, "agent did not move"
    return "agent moved"


def _step_all(names) -> str:
    import numpy as np
    from baselines.drivers import make_driver
    from envs.make_env import make_env
    times = []
    for name in names:
        env = make_env(name)
        driver = make_driver("auto", env)
        obs, _ = env.reset(seed=10_000)
        driver.reset()
        start = time.time()
        for _ in range(30):
            obs, _, terminated, truncated, _ = env.step(driver.act(obs))
            if terminated or truncated:
                obs, _ = env.reset()
                driver.reset()
        times.append((time.time() - start) / 30 * 1000)
        env.close()
    return f"{len(names)} scenarios, {np.mean(times):.0f} ms per step on average"


def rooms_run():
    from envs.make_env import ROOM_SCENARIOS
    return _step_all(list(ROOM_SCENARIOS))


def floorplans_run():
    from envs.make_env import FLOORPLAN_SCENARIOS
    return _step_all(list(FLOORPLAN_SCENARIOS))


def assistant_runs():
    from assist.driver import MODEL_PATH, AssistDriver
    from envs.make_env import make_env
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH.relative_to(ROOT)} is missing; pull it from Git, "
                                "or train one with  python training/train_assist.py")
    env = make_env("fp_office")
    driver = AssistDriver(env)
    obs, _ = env.reset(seed=10_000)
    driver.reset()
    for _ in range(30):
        obs, _, terminated, truncated, _ = env.step(driver.act(obs))
        if terminated or truncated:
            break
    env.close()
    return f"{MODEL_PATH.name} loaded and drove the chair"


def seeds_repeat():
    from envs.make_env import make_env
    env = make_env("s3_concourse_mid")
    positions = []
    for _ in range(2):
        env.reset(seed=12_345)
        u = env.unwrapped
        positions.append([(round(h.x, 4), round(h.y, 4)) for h in u.dynamic_humans] + [(round(u.robot.x, 4), round(u.robot.y, 4))])
    env.close()
    assert positions[0] == positions[1], "the same seed gave different scenes"
    return "same seed gives the same scene"


def main() -> None:
    print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}\n")
    results = [
        check("numpy", version("numpy")),
        check("gymnasium", version("gymnasium")),
        check("pygame", version("pygame")),
        check("opencv", version("cv2")),
        check("shapely", version("shapely")),
        check("torch", version("torch")),
        check("pandas + pyarrow", lambda: f"{version('pandas')()}, {version('pyarrow')()}"),
        check("stable-baselines3", version("stable_baselines3")),
        check("Python-RVO2", rvo2_works),
        check("SocNavGym", version("socnavgym")),
        check("room scenarios step", rooms_run),
        check("floor-plan scenarios step", floorplans_run),
        check("learned assistant", assistant_runs),
        check("seeds are repeatable", seeds_repeat),
    ]
    print()
    if all(results):
        print("All good. Try:  python scripts/demo.py   or   python scripts/demo.py --scenario fp_hospital")
    else:
        print("Something is missing. See the FAIL lines above, then run  bash setup.sh  again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
