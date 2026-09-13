"""Freeze the fine-tuning experiment before official CARRADA Validation is opened.

Hashes the split, the label/pos_weight record, the fine-tuning configuration, all eight fine-tuned
checkpoints, the eight frozen thresholds and every script, then writes results/freeze.json.

Refuses to overwrite an existing freeze without --refreeze, so it cannot be quietly redone once a
Validation number exists.

    python freeze.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
OUT = os.path.join(EXP, "results", "freeze.json")

LISTED = [
    "configs/internal_split.json",
    "configs/label_and_posweight.json",
    "configs/thresholds_ft.json",
]


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
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
        raise SystemExit("%s exists; a freeze happens once. Pass --refreeze only deliberately."
                         % OUT)

    files, missing = {}, []
    for rel in LISTED:
        p = os.path.join(EXP, rel)
        (files.__setitem__(rel, {"sha256": sha(p), "bytes": os.path.getsize(p)})
         if os.path.isfile(p) else missing.append(rel))
    for s in range(8):
        rel = "checkpoints/tinycnn_ft_s%d.pt" % s
        p = os.path.join(EXP, rel)
        (files.__setitem__(rel, {"sha256": sha(p), "bytes": os.path.getsize(p)})
         if os.path.isfile(p) else missing.append(rel))
    for n in sorted(os.listdir(os.path.join(EXP, "scripts"))):
        if n.endswith(".py"):
            rel = "scripts/" + n
            files[rel] = {"sha256": sha(os.path.join(EXP, rel)),
                          "bytes": os.path.getsize(os.path.join(EXP, rel))}
    for s in range(8):
        rel = "results/finetune_s%d.json" % s
        p = os.path.join(EXP, rel)
        if os.path.isfile(p):
            files[rel] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
        else:
            missing.append(rel)
    if missing:
        raise SystemExit("missing:\n  " + "\n  ".join(missing))

    split = json.load(open(os.path.join(EXP, "configs", "internal_split.json"), encoding="utf-8"))
    thr = json.load(open(os.path.join(EXP, "configs", "thresholds_ft.json"), encoding="utf-8"))
    lab = json.load(open(os.path.join(EXP, "configs", "label_and_posweight.json"),
                         encoding="utf-8"))

    roll = hashlib.sha256("\n".join("%s  %s" % (v["sha256"], k)
                                    for k, v in sorted(files.items())).encode()).hexdigest()
    out = {
        "what": "Frozen specification of the CARRADA real-data fine-tuning follow-up, hashed "
                "before official CARRADA Validation was opened.",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(files),
        "rolling_sha256": roll,
        "files": files,
        "fit_sequences": split["fit_sequences"],
        "dev_sequences": split["dev_sequences"],
        "pos_weight": lab["pos_weight"],
        "thresholds": thr["primary_thresholds"],
        "ca_dev_burden": thr["ca_cfar"]["dev_burden"],
        "carrada_validation_opened_before_freeze": False,
        "carrada_test_opened": False,
        "nothing_changes_after_validation": (
            "The eight checkpoints, the eight thresholds, the split, the label definition and the "
            "pos_weight are all hashed above. The evaluation script reads them and cannot alter "
            "them; no re-tuning, re-selection or re-thresholding happens after Validation is "
            "opened."),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("froze %d files, rolling %s" % (len(files), roll[:32]))
    print("  pos_weight      %g" % lab["pos_weight"])
    print("  CA dev burden   %.3f" % thr["ca_cfar"]["dev_burden"])
    print("  thresholds      %s" % {k: round(v, 4) for k, v in thr["primary_thresholds"].items()})
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
