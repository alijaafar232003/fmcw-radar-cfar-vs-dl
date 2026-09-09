"""Phase 6e - isolated copies of the three attribution experiments, at the revised NMS radius.

Chapter 6's experiments (E2 integrated CFAR, E3 label ablation, E4 guard masking) all extracted
candidates at radius 5. This makes patched copies that run the SAME analysis at the revised radius,
so the chapter's numbers can be checked rather than assumed.

Nothing is retrained. Every checkpoint is read from its original experiment folder; only candidate
extraction and the statistics computed from it are redone. All outputs, including the peak caches,
are written inside the e6 folder, so the original results and caches keep their radius-5 values.

Two classes of change, exactly as in `p4_patch_carrada.py`:

  1. The NMS constant is set to the audited radius.
  2. `HERE` is pinned to the ORIGINAL experiment directory (these copies live elsewhere, so the
     script's own `__file__` derivation would find neither the checkpoints nor the source), and
     the result/cache directories are then redirected into e6.
"""
from __future__ import annotations

import os
import re
import sys

E6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(E6))
DST = os.path.join(E6, "scripts", "attribution")
RADIUS = int(os.environ.get("E6_NMS_RADIUS", "2"))

# name -> (source path, original experiment dir, tag)
JOBS = {
    "e2": (os.path.join(ROOT, "experiments", "integrated_cfar", "run_integrated_cfar.py"),
           os.path.join(ROOT, "experiments", "integrated_cfar"), "e2"),
    "e3": (os.path.join(ROOT, "experiments", "label_ablation", "run_eval.py"),
           os.path.join(ROOT, "experiments", "label_ablation"), "e3"),
    "e4": (os.path.join(ROOT, "experiments", "e4_cut_guard_mask", "scripts", "evaluate_e4.py"),
           os.path.join(ROOT, "experiments", "e4_cut_guard_mask"), "e4"),
}


def patch(text, here, tag):
    e6 = E6.replace("\\", "/")
    here = here.replace("\\", "/")

    # Pin HERE / ROOT after their original definitions, before anything derived from them.
    anchor = re.search(r"^ROOT = .*$", text, flags=re.M)
    if not anchor:
        raise SystemExit("%s: no ROOT definition" % tag)
    header = (
        '\n# ------------------------------------------------- E6 PATCH (auto-generated)\n'
        '# Isolated copy at NMS radius %d. The original is not modified and keeps its radius-5\n'
        '# results. Checkpoints and source are read from the original experiment; every write\n'
        '# goes into the e6 folder.\n'
        'from pathlib import Path as _P\n'
        'HERE = _P("%s")\n'
        'ROOT = _P("%s")\n'
        'E6_RES = _P("%s") / "results" / "p10_attribution" / "%s"\n'
        'E6_RES.mkdir(parents=True, exist_ok=True)\n'
        '# ---------------------------------------------------------------------------\n'
        % (RADIUS, here, ROOT.replace("\\", "/"), e6, tag))
    text = text[:anchor.end()] + "\n" + header + text[anchor.end():]

    # Radius.
    text = re.sub(r"^NMS = \d+\s*$", "NMS = %d          # E6 PATCH" % RADIUS, text, flags=re.M)
    text = re.sub(r"^(SHAPE, NMS, TARGET_PFA, N_BOOT = \(256, 64\), )\d+", r"\g<1>%d" % RADIUS,
                  text, flags=re.M)

    # E3 sweeps four epochs to draw an epoch curve. Only the reported epoch matters for the
    # NMS check, and scoring the other three would quadruple an already substantial inference
    # cost for a diagnostic the audit does not need. The epoch curve is therefore NOT reproduced
    # here, and the original remains the record of it.
    if tag == "e3":
        text = re.sub(r"^EPOCHS = \(5, 10, 15, 21\)\s*$",
                      "EPOCHS = (21,)          # E6 PATCH: reported epoch only", text, flags=re.M)

    # ---------------------------------------------------------------- radius-matched baselines
    # E2 and E3 both compare against CACHED plain-CFAR (and, for E2, cached TinyCNN) peaks that
    # were extracted at the superseded radius. Leaving those in place would compare radius-2
    # detections against radius-5 baselines -- a mixed experiment that looks like a result. They
    # are redirected to the radius-matched caches built by `p6_final_peaks.py --split bench2_val`.
    text = text.replace(
        'base = {k: load_peaks(CV / f"peaks_{k}.npz") for k in KINDS}',
        'R2PK = _P("%s") / "results" / "final" / "peaks"          # E6 PATCH\n'
        '    base = {k: load_peaks(R2PK / ("bench2_val_r%d_%%s.npz" %% k)) for k in KINDS}'
        % (E6.replace("\\", "/"), RADIUS))
    text = text.replace(
        'dl = [load_peaks(p) for p in sorted((ROOT / "results" / "dl").glob(\n'
        '        "peaks_val_frozen_s*_ep021.npz"))]',
        'dl = [load_peaks(p) for p in sorted(R2PK.glob("bench2_val_r%d_DL_s*.npz"))]' % RADIUS)
    text = text.replace(
        '            pk = load_peaks(CV / f"peaks_{kind}.npz")',
        '            pk = load_peaks(_P("%s") / "results" / "final" / "peaks" /\n'
        '                            ("bench2_val_r%d_%%s.npz" %% kind))          # E6 PATCH'
        % (E6.replace("\\", "/"), RADIUS))

    # Results and peak caches -> e6. Reads of checkpoints (HERE / "checkpoints") are left alone.
    text = re.sub(r'^RES = HERE / "results"\s*$', "RES = E6_RES", text, flags=re.M)
    text = re.sub(r'^PK = HERE / "cache" / "peaks"\s*$',
                  'PK = E6_RES / "peaks"\nPK.mkdir(parents=True, exist_ok=True)',
                  text, flags=re.M)
    text = re.sub(r'^PKD = HERE / "cache" / "peaks"\s*$',
                  'PKD = E6_RES / "peaks"\nPKD.mkdir(parents=True, exist_ok=True)',
                  text, flags=re.M)
    return text


def main():
    os.makedirs(DST, exist_ok=True)
    which = sys.argv[1:] or list(JOBS)
    for tag in which:
        src, here, t = JOBS[tag]
        s = open(src, encoding="utf-8").read()
        out = patch(s, here, t)
        for must in ("E6_RES", "NMS = %d" % RADIUS if "NMS = " in s else "NMS"):
            if must not in out:
                raise SystemExit("%s: patch incomplete (%s)" % (tag, must))
        name = "%s_%s" % (tag, os.path.basename(src))
        with open(os.path.join(DST, name), "w", encoding="utf-8") as f:
            f.write(out)
        print("  patched %s -> %s  (radius %d)" % (tag, name, RADIUS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
