"""Isolation manifest for the REAL-DATA FINE-TUNING follow-up.

Same machinery as the CARRADA-informed experiment, with that experiment's own outputs added to
the protected set: its 8 checkpoints, its frozen config, its thresholds and its CARRADA-test
result must all survive this one untouched.

The follow-up experiment trains new models and writes new datasets. The guarantee that matters is
that none of it disturbs the frozen benchmark, the attribution experiments, the Chapter 7 result or
the original synthetic data. That guarantee is worth nothing as an assertion, so it is measured:
every protected file is hashed BEFORE anything runs and re-hashed afterwards.

    python isolation.py build     # before Phase 7
    python isolation.py verify    # after Phase 9, and after anything else that writes

Protected, by directory:
  results/dl/                      the 8 frozen checkpoints and their configs/histories
  data/bench2_*/                    the original synthetic splits (metadata + caches; frame
                                    contents are covered by a count and a sampled hash)
  configs/                          frozen_dl.yaml, final_protocol.yaml, the bench2 configs
  src/                              the simulator, detectors, evaluation and model code
  experiments/integrated_cfar/      E2
  experiments/label_ablation/       E3
  experiments/e4_cut_guard_mask/    E4
  experiments/r_carrada_real_data_validation/   R0 + R1
  experiments/e6_nms_audit/         the NMS audit and the final radius-2 artefacts
  report/                            the thesis sources and numbers.tex

Frame-level data is large, so per-split directories are summarised by file count, total bytes and
a hash over a deterministic 1-in-50 sample of frame files. That detects truncation, regeneration
and in-place edits; it is not a full byte-for-byte hash of 40 GB of frames, and says so.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
OUT = os.path.join(EXP, "results", "isolation_manifest.json")

# Hashed file by file.
FULL_DIRS = [
    "results/dl",
    "configs",
    "src",
    "experiments/integrated_cfar",
    "experiments/label_ablation",
    "experiments/e4_cut_guard_mask",
    "experiments/r_carrada_real_data_validation",
    "experiments/e6_nms_audit",
    "experiments/carrada_informed_domain",
    "report",
]
# Summarised (count + bytes + sampled hash), because they hold tens of thousands of frame files.
SAMPLED_DIRS = [
    "data/bench2_train",
    "data/bench2_val",
    "data/bench2_test",
    "data/bench2_tune",
    "data/bench2_snr_sweep",
    "data/carrada_informed_train",
    "data/carrada_informed_val",
    "data/carrada_informed_pilot",
]
SAMPLE_EVERY = 50
SKIP_PARTS = ("__pycache__", ".pytest_cache", ".ipynb_checkpoints")
SKIP_EXT = (".pyc", ".pyo")


def _walk(rel):
    base = os.path.join(ROOT, rel)
    if not os.path.isdir(base):
        return
    for root, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_PARTS]
        for n in sorted(names):
            if n.endswith(SKIP_EXT):
                continue
            p = os.path.join(root, n)
            yield p, os.path.relpath(p, ROOT).replace("\\", "/")


def _sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build():
    t0 = time.perf_counter()
    files, summaries = {}, {}

    for rel in FULL_DIRS:
        n = 0
        for p, r in _walk(rel):
            files[r] = {"sha256": _sha(p), "bytes": os.path.getsize(p)}
            n += 1
        print("  %-48s %5d files hashed" % (rel, n))

    for rel in SAMPLED_DIRS:
        allf = sorted(r for _, r in _walk(rel))
        if not allf:
            print("  %-48s absent" % rel)
            continue
        total = sum(os.path.getsize(os.path.join(ROOT, r)) for r in allf)
        pick = allf[::SAMPLE_EVERY]
        h = hashlib.sha256()
        for r in pick:
            h.update(_sha(os.path.join(ROOT, r)).encode())
            h.update(r.encode())
        summaries[rel] = {"n_files": len(allf), "total_bytes": total,
                          "n_sampled": len(pick), "sample_every": SAMPLE_EVERY,
                          "sampled_sha256": h.hexdigest()}
        print("  %-48s %6d files, %7.2f GB, %d sampled"
              % (rel, len(allf), total / 1e9, len(pick)))

    roll = hashlib.sha256(
        "\n".join("%s  %s" % (v["sha256"], k) for k, v in sorted(files.items())).encode()
    ).hexdigest()

    man = {
        "what": "Files the real-data fine-tuning follow-up experiment must not change.",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "full_dirs": FULL_DIRS,
        "sampled_dirs": SAMPLED_DIRS,
        "n_files_hashed": len(files),
        "rolling_sha256": roll,
        "files": files,
        "summaries": summaries,
        "caveat": "Sampled directories are checked by file count, total bytes and a hash over a "
                  "1-in-%d deterministic sample, not byte-for-byte over every frame."
                  % SAMPLE_EVERY,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print("\n%d files hashed, rolling sha256 %s" % (len(files), roll[:16]))
    print("%.1f s -> %s" % (time.perf_counter() - t0, os.path.relpath(OUT, ROOT)))
    return man


def verify():
    if not os.path.exists(OUT):
        raise SystemExit("no manifest at %s -- run `isolation.py build` first" % OUT)
    man = json.load(open(OUT, encoding="utf-8"))
    old = man["files"]

    changed, missing, added = [], [], []
    now = {}
    for rel in man["full_dirs"]:
        for p, r in _walk(rel):
            now[r] = {"sha256": _sha(p), "bytes": os.path.getsize(p)}

    for r, v in old.items():
        if r not in now:
            missing.append(r)
        elif now[r]["sha256"] != v["sha256"]:
            changed.append(r)
    for r in now:
        if r not in old:
            added.append(r)

    sum_changed = []
    for rel, s in man.get("summaries", {}).items():
        allf = sorted(x for _, x in _walk(rel))
        total = sum(os.path.getsize(os.path.join(ROOT, x)) for x in allf)
        pick = allf[::s["sample_every"]]
        h = hashlib.sha256()
        for x in pick:
            h.update(_sha(os.path.join(ROOT, x)).encode())
            h.update(x.encode())
        if (len(allf) != s["n_files"] or total != s["total_bytes"]
                or h.hexdigest() != s["sampled_sha256"]):
            sum_changed.append({
                "dir": rel,
                "n_files": [s["n_files"], len(allf)],
                "total_bytes": [s["total_bytes"], total],
                "sampled_sha_match": h.hexdigest() == s["sampled_sha256"]})

    print("manifest built %s" % man["built_utc"])
    print("  protected files       %d" % len(old))
    print("  MODIFIED              %d" % len(changed))
    print("  MISSING               %d" % len(missing))
    print("  added (new, allowed)  %d" % len(added))
    print("  sampled dirs changed  %d" % len(sum_changed))
    for r in changed[:25]:
        print("    MODIFIED: %s" % r)
    for r in missing[:25]:
        print("    MISSING : %s" % r)
    for r in added[:25]:
        print("    added   : %s" % r)
    for s in sum_changed:
        print("    SAMPLED DIR CHANGED: %s" % json.dumps(s))

    res = {"verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "manifest_built_utc": man["built_utc"],
           "n_protected": len(old), "modified": changed, "missing": missing,
           "added": added, "sampled_dirs_changed": sum_changed,
           "clean": not (changed or missing or sum_changed)}
    p = os.path.join(EXP, "results", "isolation_verified.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("\n%s" % ("CLEAN - nothing protected was modified"
                    if res["clean"] else "*** PROTECTED FILES CHANGED ***"))
    print("wrote %s" % os.path.relpath(p, ROOT))
    return 0 if res["clean"] else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "verify":
        sys.exit(verify())
    else:
        raise SystemExit("usage: isolation.py [build|verify]")
