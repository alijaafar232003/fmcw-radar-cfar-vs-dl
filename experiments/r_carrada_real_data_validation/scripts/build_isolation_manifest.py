"""E5 isolation manifest: SHA-256 every scientifically load-bearing asset of the frozen
synthetic benchmark, so any later modification is detectable.

READ-ONLY with respect to the project. Writes exactly one file: the manifest path given
on the command line.

Usage:
    python build_isolation_manifest.py <out.json>           # build
    python build_isolation_manifest.py --verify <ref.json>  # re-hash and compare
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# (category, [glob patterns relative to ROOT])
PROTECTED = [
    ("frozen_dl_checkpoint",
     ["results/dl/tinycnn_frozen_s{}_ep021.pt".format(i) for i in range(8)]),
    ("frozen_dl_run_config",
     ["results/dl/config_frozen_s{}.json".format(i) for i in range(8)]
     + ["results/dl/history_frozen_s{}.json".format(i) for i in range(8)]),
    ("frozen_dl_config", ["configs/frozen_dl.yaml"]),
    ("dl_source", ["src/dl/*.py"]),
    ("cfar_source", ["src/cfar/*.py"]),
    ("cfar_alpha_table", ["configs/cfar_alpha_table.json"]),
    ("eval_source", ["src/eval/*.py"]),
    ("radar_source", ["src/radar/*.py"]),
    ("data_source", ["src/data/*.py"]),
    ("dataset_config", ["configs/bench2_*.yaml"]),
    ("dataset_provenance",
     ["data/bench2_*/dataset_config_used.yaml",
      "data/bench2_*/radar_config_used.json",
      "data/bench2_*/metadata.csv",
      "data/bench2_*/axes.npz"]),
    ("dataset_nn_cache",
     ["data/bench2_*/nn_cache_x.npy", "data/bench2_*/nn_cache_y.npy"]),
    ("headline_result",
     ["results/dl/bootstrap_frozen_ep021*.csv",
      "results/dl/bootstrap_reselect_frozen_ep021*.csv",
      "results/dl/localization_frozen_ep021*.csv",
      "results/dl/localization_common_frozen_ep021*.csv",
      "results/dl/measured_pfa_frozen_ep021*.csv",
      "results/dl/runtime_cost_frozen_ep021*.csv",
      "results/dl/runtime_cost_frozen_ep021*.json",
      "results/dl/pd_vs_snr_frozen_ep021*.csv"]),
    ("cfar_baseline_result",
     ["results/cfar_test/*", "results/cfar_val/*", "results/cfar_snr_sweep/*",
      "results/cfar_alpha_audit.csv", "results/cfar_alpha_curve.csv",
      "results/psf_profile.csv"]),
    ("experiment_e2_output", ["experiments/integrated_cfar/**/*"]),
    ("experiment_e3_output", ["experiments/label_ablation/**/*"]),
    ("experiment_e4_output", ["experiments/e4_cut_guard_mask/**/*"]),
    ("preregistration", ["docs/test_preregistration.md"]),
    ("project_doc",
     ["docs/*.md", "PLAN.md", "NEXT_SESSION.md", "SCHEDULE.md", "README.md",
      "configs/*.yaml"]),
    ("project_record_append_only", ["PROGRESS.md"]),
]

EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", "venv", ".git"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    seen = {}
    out = []
    for category, pats in PROTECTED:
        for pat in pats:
            for p in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)):
                if not os.path.isfile(p):
                    continue
                rel = os.path.relpath(p, ROOT).replace("\\", "/")
                if any(part in EXCLUDE_PARTS for part in rel.split("/")):
                    continue
                if rel in seen:
                    # first category wins, so the manifest stays one row per file
                    continue
                seen[rel] = category
                out.append({"path": rel,
                            "sha256": sha256(p),
                            "size_bytes": os.path.getsize(p),
                            "category": category})
    return out


def git(*args):
    try:
        return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as exc:
        return "<unavailable: {}>".format(exc)


def environment():
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "torch", "scipy", "pandas", "yaml", "matplotlib"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            env[mod] = "not installed"
    return env


def build():
    files = collect()
    cats = {}
    for f in files:
        cats[f["category"]] = cats.get(f["category"], 0) + 1
    return {
        "experiment": "CARRADA Real-Data External Validation",
        "phase": "Phase 1 - isolation and setup",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "commit": git("rev-parse", "HEAD"),
                "status_porcelain": git("status", "--porcelain"),
                "head_log": git("log", "-1", "--format=%H %ad %s", "--date=short")},
        "environment": environment(),
        "protected_file_count": len(files),
        "protected_bytes": sum(f["size_bytes"] for f in files),
        "category_counts": cats,
        "files": files,
    }


APPEND_ONLY = "project_record_append_only"


def insertion_only_vs_git(rel):
    """True if the working copy of `rel` is the git HEAD version plus inserted lines only.

    Returns (ok, n_added, detail). Order-preserving subsequence test: every HEAD line must
    still appear, in the same order. Deleting or editing a line breaks it.
    """
    head = subprocess.run(["git", "show", "HEAD:" + rel], cwd=ROOT,
                          capture_output=True, timeout=120)
    if head.returncode != 0:
        return None, 0, "not tracked at HEAD"
    old_lines = head.stdout.decode("utf-8", "replace").splitlines()
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        new_lines = f.read().splitlines()
    i = 0
    for line in new_lines:
        if i < len(old_lines) and line == old_lines[i]:
            i += 1
    ok = i == len(old_lines)
    return ok, len(new_lines) - len(old_lines), "{} -> {} lines".format(
        len(old_lines), len(new_lines))


def verify(ref_path):
    ref = json.load(open(ref_path))
    old = {f["path"]: f for f in ref["files"]}
    new = {f["path"]: f for f in collect()}
    changed = [p for p in old if p in new and old[p]["sha256"] != new[p]["sha256"]]
    missing = [p for p in old if p not in new]
    added = [p for p in new if p not in old]

    frozen_changed = [p for p in changed if old[p]["category"] != APPEND_ONLY]
    record_changed = [p for p in changed if old[p]["category"] == APPEND_ONLY]

    print("reference manifest : {}".format(ref_path))
    print("files in reference : {}".format(len(old)))
    print("files re-hashed now: {}".format(len(new)))
    print("")
    print("FROZEN FILES CHANGED : {}   <- must be 0".format(len(frozen_changed)))
    for p in frozen_changed:
        print("   VIOLATION", p, "[{}]".format(old[p]["category"]))
    print("MISSING              : {}".format(len(missing)))
    for p in missing:
        print("   MISSING", p)
    print("NEW IN PROTECTED SET : {}".format(len(added)))
    for p in added:
        print("   NEW    ", p)

    print("APPEND-ONLY RECORDS CHANGED : {}   (expected; project convention)".format(
        len(record_changed)))
    append_ok = True
    for p in record_changed:
        ok, n_added, detail = insertion_only_vs_git(p)
        print("   {}  insertion-only vs HEAD: {}  (+{} lines, {})".format(
            p, ok, n_added, detail))
        if ok is not True:
            append_ok = False

    ok = not (frozen_changed or missing) and append_ok
    print("\nISOLATION INTACT" if ok else "\nISOLATION VIOLATED - STOP")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--verify":
        raise SystemExit(verify(sys.argv[2]))
    out = sys.argv[1]
    man = build()
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    print("protected files : {}".format(man["protected_file_count"]))
    print("protected bytes : {:.1f} MB".format(man["protected_bytes"] / 1e6))
    print("git commit      : {}  branch {}".format(man["git"]["commit"][:12],
                                                   man["git"]["branch"]))
    print("git status      : {}".format(
        "CLEAN" if not man["git"]["status_porcelain"] else "DIRTY"))
    for c, n in sorted(man["category_counts"].items()):
        print("   {:30s} {:5d}".format(c, n))
    print("written -> {}".format(out))
