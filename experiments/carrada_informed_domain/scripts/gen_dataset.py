"""Generate a CARRADA-informed synthetic split. Phases 5 and 7.

Installs the background intervention (`carrada_sim.install`) and then calls the project's own
`data.create_dataset`, unmodified, so target generation, labelling, the strata, the metadata schema
and the on-disk layout are exactly the originals.

    python gen_dataset.py --split pilot
    python gen_dataset.py --split train
    python gen_dataset.py --split val

Per-frame generator parameters (look count, whether a ridge was drawn, its CNR and texture shape)
are recorded and joined onto the dataset's own metadata.csv as `ci_*` columns, so every frame's
background is reconstructable after the fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import carrada_sim as cs                                                 # noqa: E402
from data.create_dataset import create_dataset                           # noqa: E402


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("pilot", "train", "val"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    cfg_path = os.path.join(EXP, "configs", a.split + ".yaml")
    dc = cs.DomainConfig.from_train_derived()

    print("=" * 78)
    print("CARRADA-INFORMED GENERATION -- %s" % a.split)
    print("=" * 78)
    cs.selfcheck()
    print("")
    print("domain config:")
    for k, v in dc.describe().items():
        print("  %-22s %s" % (k, v))
    print("")

    t0 = time.perf_counter()
    prev = cs.install(dc)
    try:
        create_dataset(cfg_path, overwrite=a.overwrite)
    finally:
        cs.uninstall(prev)
    secs = time.perf_counter() - t0

    # ---------------- join the per-frame generator parameters onto the dataset metadata
    rec = cs.drain_record()
    name = yaml.safe_load(open(cfg_path, encoding="utf-8"))["dataset"]["name"]
    ddir = os.path.join(ROOT, "data", name)
    mdp = os.path.join(ddir, "metadata.csv")
    md = pd.read_csv(mdp)
    if len(rec) == len(md):
        R = pd.DataFrame(rec).add_prefix("ci_")
        for c in R.columns:
            md[c] = R[c].values
        md.to_csv(mdp, index=False)
        print("joined %d per-frame background records onto metadata.csv" % len(rec))
    else:
        print("WARNING: %d records for %d frames -- background parameters NOT joined"
              % (len(rec), len(md)))

    manifest = {
        "split": a.split,
        "dataset_name": name,
        "config": os.path.relpath(cfg_path, ROOT).replace("\\", "/"),
        "config_sha256": sha256(cfg_path),
        "generator_sha256": sha256(os.path.join(HERE, "carrada_sim.py")),
        "train_targets_sha256": sha256(os.path.join(EXP, "domain_audit",
                                                    "train_derived_targets.json")),
        "domain_config": dc.describe(),
        "n_frames": int(len(md)),
        "runtime_s": round(secs, 1),
        "carrada_frames_read": 0,
        "carrada_test_read": False,
        "carrada_validation_read": False,
        "note": "Background parameters are CARRADA-TRAIN-derived. No real frame is read by this "
                "script; it consumes only the already-frozen train_derived_targets.json.",
    }
    os.makedirs(os.path.join(EXP, "results"), exist_ok=True)
    mp = os.path.join(EXP, "results", "gen_manifest_%s.json" % a.split)
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print("")
    print("%d frames in %.1f min  (%.2f s/frame)" % (len(md), secs / 60, secs / max(1, len(md))))
    print("wrote", os.path.relpath(mp, ROOT))
    if "ci_looks" in md:
        print("looks drawn:", md.ci_looks.value_counts().to_dict())
        print("ridge present on %.1f%% of frames" % (100 * md.ci_ridge.mean()))


if __name__ == "__main__":
    main()
