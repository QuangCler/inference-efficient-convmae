#!/usr/bin/env python3
"""Prepare the SCface dataset into an ImageFolder for the identity task.

SCface raw layout (after extracting official SCface zip):
    <raw>/
        mugshot_frontal_original_all/  ← high-res mugshot (enrolment/gallery)
            001.jpg ... 130.jpg
        surveillance_cameras_all/      ← low-res surveillance (probe)
            cam1_img1/ ... cam5_img3/
                001.jpg ... 130.jpg
        SCface_database_readme.pdf

Convention:
    - 130 subjects, IDs 001..130
    - 1 frontal mugshot per subject (gallery)
    - 5 cameras × 3 distances × 130 subjects (probe)

For the identity task we treat each subject folder as a class.
We use the frontal mugshots + distance-1 surveillance images for training
and distance-2 and distance-3 for validation (more realistic generalisation).

Produces (ImageFolder format):
    <out>/train/<subject_id:03d>/*.jpg    mugshots + cam*_img1
    <out>/val/<subject_id:03d>/*.jpg      cam*_img2 + cam*_img3

Usage:
    python scripts/prepare_scface.py --raw data/_raw/scface --out data/scface
"""
import argparse
import glob
import os
import re
import shutil


def _find_image_root(raw: str, name_hint: str) -> str | None:
    """Walk raw looking for a directory whose name contains name_hint."""
    for dirpath, dirnames, _ in os.walk(raw):
        for d in dirnames:
            if name_hint.lower() in d.lower():
                return os.path.join(dirpath, d)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/_raw/scface")
    ap.add_argument("--out", default="data/scface")
    args = ap.parse_args()

    raw = os.path.abspath(args.raw)
    out = os.path.abspath(args.out)

    # Locate mugshot folder
    mugshot_dir = _find_image_root(raw, "mugshot_frontal")
    if mugshot_dir is None:
        mugshot_dir = _find_image_root(raw, "mugshot")
    if mugshot_dir is None:
        raise FileNotFoundError(
            f"Cannot find mugshot directory under {raw}. "
            "Expected: mugshot_frontal_original_all/ or similar."
        )

    # Locate surveillance folder
    surv_dir = _find_image_root(raw, "surveillance_cameras")
    if surv_dir is None:
        surv_dir = _find_image_root(raw, "surveillance")
    if surv_dir is None:
        raise FileNotFoundError(
            f"Cannot find surveillance directory under {raw}. "
            "Expected: surveillance_cameras_all/ or similar."
        )

    subj_re = re.compile(r"(\d{3})[_-].*\.(jpg|JPG|bmp|BMP|png)$")

    # ------ Gallery / mugshots → train ------
    n_train = n_val = 0
    for fpath in sorted(glob.glob(os.path.join(mugshot_dir, "*"))):
        m = subj_re.search(os.path.basename(fpath))
        if not m:
            continue
        subj_id = m.group(1)
        dst_dir = os.path.join(out, "train", subj_id)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(fpath, os.path.join(dst_dir, "mugshot.jpg"))
        n_train += 1

    # ------ Surveillance probes ------
    # cam*_img1 → train, cam*_img2 / cam*_img3 → val
    for cam_dir in sorted(glob.glob(os.path.join(surv_dir, "*"))):
        if not os.path.isdir(cam_dir):
            continue
        cam_name = os.path.basename(cam_dir)
        # distance label: img1 = distance 1 (closest), img2/3 = farther
        if "img1" in cam_name:
            split = "train"
        else:
            split = "val"
        for fpath in sorted(glob.glob(os.path.join(cam_dir, "*"))):
            m = subj_re.search(os.path.basename(fpath))
            if not m:
                continue
            subj_id = m.group(1)
            dst_dir = os.path.join(out, split, subj_id)
            os.makedirs(dst_dir, exist_ok=True)
            dst_fname = f"{cam_name}_{os.path.basename(fpath)}"
            shutil.copy2(fpath, os.path.join(dst_dir, dst_fname))
            if split == "train":
                n_train += 1
            else:
                n_val += 1

    # Count unique subjects
    subjects = set(
        os.path.basename(d)
        for split in ("train", "val")
        for d in glob.glob(os.path.join(out, split, "*"))
        if os.path.isdir(d)
    )
    print(f"[scface] subjects={len(subjects)}  train_imgs={n_train}  val_imgs={n_val} → {out}")


if __name__ == "__main__":
    main()
