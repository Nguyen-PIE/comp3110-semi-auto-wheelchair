"""Which Webots worlds are for training and which for testing.

Training uses seven buildings, each as it is and in its furnished copies (_v1, _v2, ...).
Testing uses what training never saw:
  - new furniture: one more furnished copy of each training building (<name>_test)
  - a new building: fp_house, which is never trained on at all (as it is, and furnished)
Trips also use different seeds (different starts, goals and people): training uses the
training seeds 0-9999, testing the validation seeds 10000-10199, as in 2D.

Make all of them with:  python webots/make_world.py all --variants 3
"""

from __future__ import annotations

from pathlib import Path

WORLDS = Path(__file__).resolve().parent / "worlds"

TRAIN_BUILDINGS = ["corridor", "fp_hospital", "fp_office", "fp_warehouse",
                   "fp_depot", "fp_warehouse2", "fp_warehouse3"]
HELD_OUT_BUILDINGS = ["fp_house"]


def made(name: str) -> bool:
    return (WORLDS / f"{name}.wbt").exists()


def train_worlds() -> list[str]:
    """Every training building and its furnished copies that have been made."""
    names = []
    for b in TRAIN_BUILDINGS:
        copies = sorted((p.stem for p in WORLDS.glob(f"{b}_v*.wbt")),
                        key=lambda n: int(n.rsplit("_v", 1)[1]) if n.rsplit("_v", 1)[1].isdigit() else 0)
        names += [n for n in [b, *copies] if made(n)]
    return names


def test_worlds() -> list[str]:
    """New furniture in the training buildings, and the held-out building."""
    names = [f"{b}_test" for b in TRAIN_BUILDINGS]
    for b in HELD_OUT_BUILDINGS:
        names += [b, f"{b}_test"]
    return names


def kind(world: str) -> str:
    """What is new in this world compared with training."""
    building = world.removesuffix("_test")
    if building in HELD_OUT_BUILDINGS:
        return "new building"
    return "new furniture" if world.endswith("_test") else "trained on"


GROUPS = {"train": train_worlds, "test": test_worlds}


def expand(names: list[str]) -> list[str]:
    """Replace the group names 'train' and 'test' by their worlds."""
    out = []
    for n in names:
        out += GROUPS[n]() if n in GROUPS else [n]
    return out
