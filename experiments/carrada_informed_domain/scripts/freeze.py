"""Freeze the experiment specification. Phase 6.

Hashes every file listed under `hashed:` in FROZEN_CONFIG.yaml, plus the dataset provenance files,
and writes results/freeze.json. Run AFTER the datasets exist and BEFORE training starts, so the
record covers the data the models were actually trained on.

Refuses to run if any listed file is missing, and refuses to overwrite an existing freeze unless
--refreeze is passed, so a freeze cannot be silently redone after a result is known.

    python freeze.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
OUT = os.path.join(EXP, "results", "freeze.json")

DATASETS = ("carrada_informed_train", "carrada_informed_val", "carrada_informed_pilot")
PROV = ("dataset_config_used.yaml", "radar_config_used.json", "metadata.csv")


def sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refreeze", action="store_true")
    a = ap.parse_args()

    if os.path.exists(OUT) and not a.refreeze:
        raise SystemExit("%s already exists. A freeze is meant to happen once; pass --refreeze "
                         "only if you intend to replace it and can say why." % OUT)

    cfgp = os.path.join(EXP, "FROZEN_CONFIG.yaml")
    cfg = yaml.safe_load(open(cfgp, encoding="utf-8"))
    listed = list(cfg["hashed"])

    files, missing = {}, []
    for rel in listed:
        p = os.path.join(EXP, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        files[rel] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
    if missing:
        raise SystemExit("listed but missing:\n  " + "\n  ".join(missing))

    # Dataset provenance: the fingerprint of the data the models will be trained on.
    ds = {}
    for name in DATASETS:
        d = os.path.join(ROOT, "data", name)
        if not os.path.isdir(d):
            ds[name] = {"present": False}
            continue
        parts = {}
        for pn in PROV:
            p = os.path.join(d, pn)
            if os.path.isfile(p):
                parts[pn] = sha(p)
        nfr = len([x for x in os.listdir(os.path.join(d, "frames"))]) \
            if os.path.isdir(os.path.join(d, "frames")) else 0
        fp = hashlib.sha256("|".join("%s:%s" % (k, v) for k, v in sorted(parts.items()))
                            .encode()).hexdigest()
        ds[name] = {"present": True, "n_frames": nfr, "provenance": parts, "fingerprint": fp}

    roll = hashlib.sha256(
        "\n".join("%s  %s" % (v["sha256"], k) for k, v in sorted(files.items())).encode()
    ).hexdigest()

    out = {
        "what": "Frozen specification of the CARRADA-informed follow-up, hashed before training "
                "and before any CARRADA test read.",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_config_sha256": files["FROZEN_CONFIG.yaml"]["sha256"],
        "pre_analysis_plan_sha256": files["pre_analysis_plan.md"]["sha256"],
        "n_files": len(files),
        "rolling_sha256": roll,
        "files": files,
        "datasets": ds,
        "carrada_test_read_before_freeze": False,
        "carrada_validation_read": False,
        "models_trained_before_freeze": 0,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("froze %d files" % len(files))
    print("  FROZEN_CONFIG.yaml     %s" % out["frozen_config_sha256"][:32])
    print("  pre_analysis_plan.md   %s" % out["pre_analysis_plan_sha256"][:32])
    print("  rolling                %s" % roll)
    for name, v in ds.items():
        print("  %-26s %s" % (name, ("%d frames, fp %s" % (v["n_frames"], v["fingerprint"][:16]))
                              if v["present"] else "ABSENT"))
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
