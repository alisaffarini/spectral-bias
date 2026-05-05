"""Sequentially download fine-grained datasets (no parallel race)."""
import os
import shutil
from pathlib import Path
import torchvision

DATA_ROOT = "./data"


def safe_remove(path: Path):
    if path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except Exception as e:
            print(f"  could not remove {path}: {e}")


# Clean any partial downloads first
data_dir = Path(DATA_ROOT)
for partial in ["fgvc-aircraft-2013b.tar.gz", "fgvc-aircraft-2013b",
                "102flowers.tgz", "flowers-102", "imagelabels.mat",
                "setid.mat", "oxford-iiit-pet"]:
    safe_remove(data_dir / partial)

print("=== Pet ===", flush=True)
torchvision.datasets.OxfordIIITPet(DATA_ROOT, split="trainval", download=True)
print("  ok")

print("=== Aircraft ===", flush=True)
torchvision.datasets.FGVCAircraft(DATA_ROOT, split="trainval", download=True)
print("  ok")

print("=== Flowers ===", flush=True)
torchvision.datasets.Flowers102(DATA_ROOT, split="train", download=True)
print("  ok")

print("all done.")
