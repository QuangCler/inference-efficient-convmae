#!/usr/bin/env python3
"""Prepare the Kaggle CelebA dataset into the layout finetune/common/data/celeba_attr.py expects.

Kaggle 'jessicali9530/celeba-dataset' raw layout:
    <raw>/img_align_celeba/img_align_celeba/*.jpg
    <raw>/list_attr_celeba.csv          # image_id + 40 attrs in {-1,+1}
    <raw>/list_eval_partition.csv       # image_id, partition (0=train,1=val,2=test)

Produces:
    <out>/images -> symlink to the jpg dir
    <out>/attr.csv       (copy of list_attr_celeba.csv; adapter maps -1->0)
    <out>/splits/train.txt, val.txt

Usage:
    python scripts/prepare_celeba.py --raw data/_raw/celeba --out data/celeba
"""
import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/_raw/celeba")
    ap.add_argument("--out", default="data/celeba")
    ap.add_argument("--val_partition", type=int, default=1,
                    help="which partition id is val (1=official val; use 2 to add test)")
    args = ap.parse_args()

    raw, out = os.path.abspath(args.raw), os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "splits"), exist_ok=True)

    # image dir (nested once in this Kaggle release)
    img_dir = os.path.join(raw, "img_align_celeba", "img_align_celeba")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(raw, "img_align_celeba")
    link = os.path.join(out, "images")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link) if os.path.islink(link) else None
    if not os.path.exists(link):
        os.symlink(img_dir, link)

    # attr.csv (adapter reads header + 40 cols, filename in col0)
    shutil.copyfile(os.path.join(raw, "list_attr_celeba.csv"), os.path.join(out, "attr.csv"))

    # splits from partition file
    train, val = [], []
    with open(os.path.join(raw, "list_eval_partition.csv"), newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if not row:
                continue
            fn, part = row[0].strip(), int(row[1])
            if part == 0:
                train.append(fn)
            elif part == args.val_partition:
                val.append(fn)
    with open(os.path.join(out, "splits", "train.txt"), "w") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(out, "splits", "val.txt"), "w") as f:
        f.write("\n".join(val) + "\n")

    print(f"[celeba] images -> {img_dir}")
    print(f"[celeba] train={len(train)} val={len(val)}  (val_partition={args.val_partition})")
    print(f"[celeba] wrote {out}/attr.csv, splits/")


if __name__ == "__main__":
    main()
