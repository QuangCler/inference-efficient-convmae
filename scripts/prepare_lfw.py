#!/usr/bin/env python3
"""Prepare the Kaggle LFW dataset into the layout finetune/common/data/lfw_verify.py expects.

Kaggle 'jessicali9530/lfw-dataset' raw layout:
    <raw>/lfw-deepfunneled/lfw-deepfunneled/<Name>/<Name>_NNNN.jpg
    <raw>/pairs.csv        # 6000 official pairs; matched: name,n1,n2  mismatched: name1,n1,name2,n2
                           #  (Windows CRLF -> strip \\r; trailing comma on matched rows)

Produces:
    <out>/images -> symlink to the <Name>/ dir root
    <out>/pairs.txt        # tab-separated, format the adapter parses

Usage:
    python scripts/prepare_lfw.py --raw data/_raw/lfw --out data/lfw
"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/_raw/lfw")
    ap.add_argument("--out", default="data/lfw")
    args = ap.parse_args()

    raw, out = os.path.abspath(args.raw), os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    # image root that directly contains <Name>/ subdirs (nested once here)
    img_root = os.path.join(raw, "lfw-deepfunneled", "lfw-deepfunneled")
    if not os.path.isdir(img_root):
        img_root = os.path.join(raw, "lfw-deepfunneled")
    link = os.path.join(out, "images")
    if os.path.islink(link):
        os.remove(link)
    if not os.path.exists(link):
        os.symlink(img_root, link)

    # pairs.csv -> pairs.txt. Interleave matched/mismatched so any prefix is class-balanced
    # (makes capped dev/smoke eval meaningful; the full-set metric is order-independent).
    matched, mismatched = [], []
    with open(os.path.join(raw, "pairs.csv"), newline="") as f:
        r = csv.reader(f)
        next(r, None)  # header: name,imagenum1,imagenum2,
        for row in r:
            cells = [c.strip() for c in row if c.strip()]  # strip \r + drop empty trailing
            if len(cells) == 3:                # matched: same person
                name, n1, n2 = cells
                matched.append(f"{name}\t{n1}\t{n2}")
            elif len(cells) == 4:              # mismatched: different people
                name1, n1, name2, n2 = cells
                mismatched.append(f"{name1}\t{n1}\t{name2}\t{n2}")

    with open(os.path.join(out, "pairs.txt"), "w") as g:
        for i in range(max(len(matched), len(mismatched))):
            if i < len(matched):
                g.write(matched[i] + "\n")
            if i < len(mismatched):
                g.write(mismatched[i] + "\n")
    print(f"[lfw] images -> {img_root}")
    print(f"[lfw] pairs.txt: matched={len(matched)} mismatched={len(mismatched)} "
          f"total={len(matched) + len(mismatched)} (interleaved)")


if __name__ == "__main__":
    main()
