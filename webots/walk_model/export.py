"""Turn a Trajectron++ training checkpoint into the walking model the simulator loads.

Training (train.sh) saves whole Python objects, which PyTorch will only load with its safety
check off. This keeps just the numbers (a "state dict"), which load safely:

    python webots/walk_model/export.py <training folder> [epoch]   ->  models/walk_trajectron/
"""
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "third_party" / "Trajectron-plus-plus" / "trajectron"))   # its classes, to load
OUT = ROOT / "models" / "walk_trajectron"


def main(argv):
    folder = Path(argv[0])
    epochs = sorted(int(p.stem.split("-")[1]) for p in folder.glob("model_registrar-*.pt"))
    epoch = int(argv[1]) if len(argv) > 1 else epochs[-1]
    # Our own training output, so it is safe to load it in full here.
    modules = torch.load(folder / f"model_registrar-{epoch}.pt", map_location="cpu", weights_only=False)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(modules.state_dict(), OUT / "walk_model.pt")
    config = json.loads((folder / "config.json").read_text())
    config["trained_epochs"] = epoch
    (OUT / "config.json").write_text(json.dumps(config, indent=1))
    print(f"Saved epoch {epoch} to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
