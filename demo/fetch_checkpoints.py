#!/usr/bin/env python3
"""Download the fine-tuned face checkpoints into ./checkpoints/ (run once on your laptop).

Uses gdown against Google Drive — works on any normal internet connection. ~5.5 GB total.
Files already present (correct size) are skipped, so it is safe to re-run.

    pip install gdown
    python fetch_checkpoints.py
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "checkpoints")
os.makedirs(DEST, exist_ok=True)

# filename -> (Google Drive id, expected md5 or None). base/ghost fine-tuned face models.
FILES = {
    "celeba_baseline.pth": ("1p-6NGTMbEPEdOvTC6BDBYOUXMQ-sNPBx", "988e751f08b9c2118ffea052099c5d7a"),
    "celeba_ghost.pth":    ("1DGjuWOIA198UjTd-eRUh-5uzcrQwldZJ", "7d6b0f8460e2640a9d5ff45cdd0d4470"),
    "lfw_baseline.pth":    ("1blN2E1ZnOCqWA6Erijzw3rM3LdF38e2A", "00d59190459e857201794bf467aa4a4a"),
    "lfw_ghost.pth":       ("1xGMSrTdYmByEr2ik99kpbbzlWxDSbAbl", "cfde11f27fac0680a187e80611caf546"),
    "casia_ghost.pth":     ("1DeQ8YNvm4GQ1n3WIzcXwgEkJ4UAHDyTp", None),
    # SCface (cross-resolution identity, 130 classes) — seed0, the best valid seed
    # (base 46.7% / ghost 31.7% Top-1). The low-res stressor where Ghost's compression
    # costs most, mirroring the report's largest accuracy gap.
    "scface_baseline.pth": ("1nBtBjt9EmEoYFF5XHtkqX8zoiQRB09T-", None),
    "scface_ghost.pth":    ("1n4wKYnRO5HiIXW_S1rVvOvbwYxEJ_MOa", None),
    # casia_baseline.pth = freshly retrained demo checkpoint (40 ep, Top-1 80.6%); the report's
    # CASIA numbers are unchanged (they use the original 3-seed runs). Uploaded to the shared
    # finetune_result/baseline/casia/outputs/seed0_demo/ folder.
    "casia_baseline.pth":  ("15zYNKaqsLjApfmMGcbQrT37BFq3pLes9", None),
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    try:
        import gdown
    except ImportError:
        print("Please `pip install gdown` first.", file=sys.stderr)
        sys.exit(1)

    ok, skipped, failed = [], [], []
    for name, (fid, want_md5) in FILES.items():
        dst = os.path.join(DEST, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            if want_md5 is None or md5(dst) == want_md5:
                print(f"[skip] {name} already present"); skipped.append(name); continue
        if fid == "FILL_ME_AFTER_UPLOAD":
            print(f"[pending] {name}: no Drive id yet (demo runs without it)."); continue
        print(f"[dl] {name} ...")
        try:
            gdown.download(id=fid, output=dst, quiet=False)
            if want_md5 and md5(dst) != want_md5:
                print(f"[warn] {name}: md5 mismatch (file may still work)")
            ok.append(name)
        except Exception as e:
            print(f"[fail] {name}: {e}"); failed.append(name)

    print(f"\nDone. downloaded={len(ok)} skipped={len(skipped)} failed={len(failed)}")
    if failed:
        print("Retry later for:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
