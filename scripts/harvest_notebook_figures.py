"""Re-execute the analysis notebooks and harvest every figure they produce, for the report.

The committed notebooks carry plotting code but no stored outputs, so the figures that
`PROGRESS.md` refers to do not exist as files. This script re-runs each notebook (in its own
directory, so relative imports and data paths resolve), extracts the embedded PNGs, and writes an
index describing what each figure is.

READ-ONLY with respect to the committed notebooks: execution output goes to a scratch copy, never
back over the original. Figures land in report/figures/harvested/.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(ROOT, "report", "_harvest_exec")
OUTDIR = os.path.join(ROOT, "report", "figures", "harvested")
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")

NOTEBOOKS = [
    "notebooks/cfar_baseline.ipynb",
    "notebooks/cnn_development_log.ipynb",
    "notebooks/evaluation_metrics_overview.ipynb",
    "notebooks/dl_development.ipynb",
    "notebooks/dataset_overview.ipynb",
    "notebooks/cfar_vs_dl_comparison.ipynb",
    "notebooks/InteractivePipeline.ipynb",
    "experiments/integrated_cfar/integrated_cfar_experiment.ipynb",
    "experiments/label_ablation/label_ablation_experiment.ipynb",
    "experiments/e4_cut_guard_mask/e4_cut_guard_mask.ipynb",
]

TIMEOUT = 2400


def clean(s):
    return re.sub(r"[^\x20-\x7e]", " ", s).strip()


def execute(rel):
    """Run one notebook and return the path to the executed copy, or None.

    The committed notebook is NEVER touched. An earlier version passed the original path to
    nbconvert with --output-dir, which still rewrote several sources in place -- including three
    notebooks protected by the CARRADA experiment's isolation manifest. They had to be restored
    from git.

    The fix: copy the notebook to a temporary sibling INSIDE its own directory, execute the copy
    (so relative imports and data paths still resolve), then delete it. The original is only ever
    read.
    """
    src = os.path.join(ROOT, rel)
    tag = os.path.splitext(os.path.basename(rel))[0]
    out = os.path.join(SCRATCH, tag + "_exec.ipynb")
    if not os.path.isfile(src):
        return None, "missing"

    nbdir = os.path.dirname(src)
    tmp = os.path.join(nbdir, "_harvest_tmp_{}.ipynb".format(tag))
    before = os.path.getmtime(src)
    shutil.copy2(src, tmp)
    cmd = [PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout={}".format(TIMEOUT),
           "--output-dir", SCRATCH, "--output", os.path.basename(out), tmp]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=nbdir)
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)
    dt = time.time() - t0

    if os.path.getmtime(src) != before:
        return None, "ABORT: source notebook was modified -- restore it from git"
    if not os.path.isfile(out):
        tail = (r.stderr or "").strip().splitlines()[-1:] or ["no output"]
        return None, "FAILED ({:.0f}s): {}".format(dt, clean(tail[0])[:80])
    return out, "{:.0f}s".format(dt)


def harvest(nb_path, tag, index):
    nb = json.load(open(nb_path, encoding="utf-8"))
    head, n = "", 0
    for ci, c in enumerate(nb.get("cells", [])):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "markdown":
            for line in src.split("\n"):
                if line.strip().startswith("#"):
                    head = clean(line.lstrip("#"))[:80]
            continue
        titles = re.findall(
            r'(?:set_title|suptitle|plt\.title)\(\s*[fr]?["\']([^"\']{4,90})', src)
        for o in c.get("outputs", []):
            png = o.get("data", {}).get("image/png")
            if not png:
                continue
            n += 1
            name = "{}_{:02d}.png".format(tag, n)
            with open(os.path.join(OUTDIR, name), "wb") as f:
                f.write(base64.b64decode(png))
            index.append({
                "file": "report/figures/harvested/" + name,
                "notebook": tag,
                "cell": ci,
                "section": head,
                "title": clean(titles[0]) if titles else "",
            })
    return n


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)
    index, summary = [], []
    for rel in NOTEBOOKS:
        tag = os.path.splitext(os.path.basename(rel))[0]
        print("executing {:52s} ".format(rel), end="", flush=True)
        path, status = execute(rel)
        if path is None:
            print(status)
            summary.append((tag, 0, status))
            continue
        n = harvest(path, tag, index)
        print("{:>8s}   {:3d} figures".format(status, n))
        summary.append((tag, n, status))

    with open(os.path.join(OUTDIR, "figure_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    try:
        import pandas as pd
        pd.DataFrame(index).to_csv(os.path.join(OUTDIR, "figure_index.csv"), index=False)
    except Exception:
        pass

    print("\n" + "=" * 78)
    print("{:44s} {:>8s}  {}".format("notebook", "figures", "status"))
    for tag, n, status in summary:
        print("{:44s} {:>8d}  {}".format(tag, n, status))
    print("-" * 78)
    print("TOTAL harvested: {} figures -> report/figures/harvested/".format(len(index)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
