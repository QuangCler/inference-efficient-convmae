#!/usr/bin/env python3
"""Download the original face datasets used by the project (into ./datasets/).

Uses the Kaggle API for the openly-downloadable sets. These are large — grab only what you
need with --only.

    LFW     jessicali9530/lfw-dataset       (~0.2 GB)   public benchmark
    CelebA  jessicali9530/celeba-dataset    (~1.4 GB)   research use
    CASIA   debarghamitraroy/casia-webface  (~4 GB)     research use
    SCface  — access-controlled — not downloadable here (request a licence; see below)

Setup once:
    pip install kaggle
    # put your token at ~/.kaggle/kaggle.json  (Kaggle -> Account -> Create New API Token)
    #   or export KAGGLE_USERNAME=... KAGGLE_KEY=...

Usage:
    python fetch_datasets.py                 # LFW + CelebA + CASIA
    python fetch_datasets.py --only lfw      # just one
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "datasets")

KAGGLE = {
    "lfw":    "jessicali9530/lfw-dataset",
    "celeba": "jessicali9530/celeba-dataset",
    "casia":  "debarghamitraroy/casia-webface",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(KAGGLE), help="subset to download")
    args = ap.parse_args()
    names = args.only or list(KAGGLE)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print("Please `pip install kaggle` and set up your Kaggle token "
              "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).", file=sys.stderr)
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"Kaggle auth failed: {e}\nCreate a token at Kaggle -> Account -> Create New API Token.",
              file=sys.stderr)
        sys.exit(1)

    for name in names:
        out = os.path.join(DEST, name)
        os.makedirs(out, exist_ok=True)
        if os.listdir(out):
            print(f"[skip] {name}: {out}/ already populated"); continue
        print(f"[dl] {name}  <-  {KAGGLE[name]}  (this can be large)")
        api.dataset_download_files(KAGGLE[name], path=out, unzip=True, quiet=False)
        print(f"[ok] {name} -> {out}/")

    print("\nSCface is access-controlled and cannot be auto-downloaded. Request a research licence at "
          "https://www.scface.org/ and place the images under datasets/scface/. The report uses SCface "
          "strictly under its licence and redistributes no personal data.")
    print(f"\nDatasets under {DEST}/  — drop any face image into the app to test.")


if __name__ == "__main__":
    main()
