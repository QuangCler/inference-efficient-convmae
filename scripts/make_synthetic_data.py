#!/usr/bin/env python3
"""Generate tiny synthetic datasets for the smoke test — NO real data needed.

Writes under finetune/_smoke_data/<dataset>/ one example per task family:
  identity  : ImageFolder (train/val, few classes x few random RGB jpgs)
  liveness  : ImageFolder with real/ + attack/ dirs
  celeba    : images/ + attr.csv (40 cols) + splits/{train,val}.txt
  lfw       : images/<person>/ + pairs.txt (tiny matched/mismatched list)

Deterministic (numpy-seeded). Run:  PYTHONPATH=./.pylibs python3 scripts/make_synthetic_data.py
"""
import os

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "finetune", "_smoke_data")
RNG = np.random.RandomState(0)
IMG = 64  # small; transforms resize to input_size anyway


def _rand_img(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = RNG.randint(0, 256, (IMG, IMG, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=90)


def make_imagefolder(base, classes, n_per_class_train=4, n_per_class_val=2):
    for split, n in (("train", n_per_class_train), ("val", n_per_class_val)):
        for c in classes:
            for i in range(n):
                _rand_img(os.path.join(base, split, c, f"{c}_{i}.jpg"))


def make_identity():
    make_imagefolder(os.path.join(ROOT, "identity"),
                     classes=[f"id{i:03d}" for i in range(6)])


def make_liveness():
    make_imagefolder(os.path.join(ROOT, "3dmad"), classes=["real", "attack"],
                     n_per_class_train=6, n_per_class_val=4)


def make_celeba():
    base = os.path.join(ROOT, "celeba")
    img_dir = os.path.join(base, "images")
    n_total = 20
    files = [f"{i:06d}.jpg" for i in range(n_total)]
    for fn in files:
        _rand_img(os.path.join(img_dir, fn))
    # attr.csv: header + 40 attr cols in {-1,1} (native CelebA encoding)
    os.makedirs(os.path.join(base, "splits"), exist_ok=True)
    header = ["filename"] + [f"attr{j:02d}" for j in range(40)]
    with open(os.path.join(base, "attr.csv"), "w") as f:
        f.write(",".join(header) + "\n")
        for fn in files:
            vals = RNG.choice([-1, 1], size=40)
            f.write(fn + "," + ",".join(str(int(v)) for v in vals) + "\n")
    with open(os.path.join(base, "splits", "train.txt"), "w") as f:
        f.write("\n".join(files[:14]) + "\n")
    with open(os.path.join(base, "splits", "val.txt"), "w") as f:
        f.write("\n".join(files[14:]) + "\n")


def make_lfw():
    base = os.path.join(ROOT, "lfw")
    img_dir = os.path.join(base, "images")
    people = [f"Person_{i}" for i in range(5)]
    for p in people:
        for i in range(1, 4):  # 3 imgs each -> all usable by the >=2 proxy
            _rand_img(os.path.join(img_dir, p, f"{p}_{i:04d}.jpg"))
    # pairs.txt: header, then matched (same person) + mismatched (diff people)
    lines = ["3\t3"]
    for p in people[:3]:            # matched
        lines.append(f"{p}\t1\t2")
    for i in range(3):              # mismatched
        a, b = people[i], people[(i + 1) % len(people)]
        lines.append(f"{a}\t1\t{b}\t2")
    with open(os.path.join(base, "pairs.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    make_identity()
    make_liveness()
    make_celeba()
    make_lfw()
    print(f"[synthetic] wrote datasets under {ROOT}")
    for d in ("identity", "3dmad", "celeba", "lfw"):
        print(f"  - {d}: {os.path.join(ROOT, d)}")


if __name__ == "__main__":
    main()
