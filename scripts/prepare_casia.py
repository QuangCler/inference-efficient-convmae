#!/usr/bin/env python3
"""Unpack the Kaggle CASIA-WebFace (InsightFace RecordIO) into an ImageFolder for the identity task.

Kaggle 'debarghamitraroy/casia-webface' ships InsightFace packing, NOT raw jpgs:
    <raw>/casia-webface/{train.rec, train.idx, train.lst, property}
    property = "<num_classes>,<H>,<W>"  (e.g. 10572,112,112)
Each record's IRHeader carries the identity label; the payload is a 112x112 aligned jpg.

This module implements a dependency-free MXNet-RecordIO reader (mxnet does not install on
py3.13), so no mxnet/insightface needed. Produces:
    <out>/train/<label:05d>/*.jpg
    <out>/val/<label:05d>/*.jpg          (val_per_class held out per identity)

Usage:
    python scripts/prepare_casia.py --raw data/_raw/casia/casia-webface --out data/casia
    python scripts/prepare_casia.py ... --limit_classes 20 --max_per_class 20   # fast subset
    python scripts/prepare_casia.py ... --probe 5                               # just verify framing
"""
import argparse
import io
import os
import struct

from PIL import Image

_MAGIC = 0xCED7230A


def _read_idx(idx_path):
    """train.idx: lines '<key>\\t<offset>'. Returns list of (key, offset) sorted by key."""
    out = []
    with open(idx_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            k, off = ln.split("\t")
            out.append((int(k), int(off)))
    out.sort()
    return out


def _read_record(fp, offset):
    """Read one MXRecordIO record at byte `offset`. Returns raw record bytes (header+payload)."""
    fp.seek(offset)
    magic, lrecord = struct.unpack("<II", fp.read(8))
    assert magic == _MAGIC, f"bad record magic {magic:#x} at offset {offset}"
    # lrecord packs a 3-bit cflag (continuation) in the top bits and length in the low 29.
    length = lrecord & ((1 << 29) - 1)
    return fp.read(length)


def _parse_record(data):
    """Split IRHeader (flag,label,id,id2) from image payload. Returns (label:int, img_bytes)."""
    flag, label, _id, _id2 = struct.unpack("<IfQQ", data[:24])
    payload = data[24:]
    if flag > 0:  # label is an array of `flag` floats right after the header
        payload = data[24 + flag * 4:]
    return int(label), payload


def _iter_records(rec_path, idx):
    with open(rec_path, "rb") as fp:
        for key, off in idx:
            data = _read_record(fp, off)
            label, img = _parse_record(data)
            yield key, label, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/_raw/casia/casia-webface")
    ap.add_argument("--out", default="data/casia")
    ap.add_argument("--val_per_class", type=int, default=2)
    ap.add_argument("--limit_classes", type=int, default=0, help="0=all")
    ap.add_argument("--max_per_class", type=int, default=0, help="0=all")
    ap.add_argument("--probe", type=int, default=0, help="just decode N records and exit")
    args = ap.parse_args()

    raw = os.path.abspath(args.raw)
    idx = _read_idx(os.path.join(raw, "train.idx"))
    rec = os.path.join(raw, "train.rec")

    if args.probe:
        n_ok = 0
        for key, label, img in _iter_records(rec, idx):
            try:
                im = Image.open(io.BytesIO(img)).convert("RGB")
            except Exception as e:
                print(f"  key={key} label={label} NOT-AN-IMAGE ({type(e).__name__}) len={len(img)}")
                continue
            print(f"  key={key} label={label} size={im.size} mode={im.mode} bytes={len(img)}")
            n_ok += 1
            if n_ok >= args.probe:
                break
        print(f"[probe] decoded {n_ok} image records OK")
        return

    out = os.path.abspath(args.out)
    per_class_count = {}
    n_written = n_val = 0
    kept_labels = set()

    for key, label, img in _iter_records(rec, idx):
        # Labels are packed in ascending order, so once we pass the class limit we are done
        # (early break avoids decoding the rest of the 494k records for a subset).
        if args.limit_classes and label >= args.limit_classes:
            break
        c = per_class_count.get(label, 0)
        if args.max_per_class and c >= args.max_per_class:
            continue
        # skip non-image header records (InsightFace stores a meta record at key 0)
        try:
            im = Image.open(io.BytesIO(img)).convert("RGB")
        except Exception:
            continue
        split = "val" if c < args.val_per_class else "train"
        d = os.path.join(out, split, f"{label:05d}")
        os.makedirs(d, exist_ok=True)
        im.save(os.path.join(d, f"{key}.jpg"), quality=95)
        per_class_count[label] = c + 1
        kept_labels.add(label)
        if split == "val":
            n_val += 1
        else:
            n_written += 1
        if (n_written + n_val) % 20000 == 0:
            print(f"  ... {n_written + n_val} images, {len(kept_labels)} classes")

    print(f"[casia] classes={len(kept_labels)} train_imgs={n_written} val_imgs={n_val} -> {out}")


if __name__ == "__main__":
    main()
