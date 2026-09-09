"""Phase 4 setup - make isolated copies of the three CARRADA R1 scripts.

The originals are manifest-protected and are NEVER edited. This produces patched copies inside
the e6 folder with exactly two classes of change:

  1. `det.NMS_SIZE` is overridden immediately after `import detectors as det`, so the frozen
     CARRADA path runs at the audited radius instead of 5. `NMS_SIZE` is read inside the detector
     functions rather than bound as a default argument, so a post-import override is sufficient
     and no CARRADA source needs forking.

  2. Output paths are redirected into the e6 folder. READS of the original experiment's frozen
     inputs (preregistration hash, CARRADA data) are left pointing at the original.

Everything else, including the thresholds-from-bench2_val-only discipline and its guards, is
untouched.
"""
from __future__ import annotations

import os
import re
import sys

E6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(E6))
SRC = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation", "scripts")
DST = os.path.join(E6, "scripts", "carrada")

RADIUS = int(os.environ.get("E6_NMS_RADIUS", "2"))

SCRIPTS = ("derive_tinycnn_thresholds.py", "run_r1_test.py", "analyze_r1_test.py",
           "build_tables_from_raw.py")

HEADER = '''
# ---------------------------------------------------------------- E6 PATCH (auto-generated)
# Isolated copy. The original lives in experiments/r_carrada_real_data_validation/scripts/ and is
# not modified. Two changes only: the NMS radius override below, and output paths redirected into
# the e6 experiment folder. Reads of frozen inputs still point at the original experiment.
E6_DIR = "{e6}"
E6_NMS_RADIUS = {radius}
# This copy lives one directory deeper than the original, so the script's own EXP/ROOT derivation
# from __file__ would point at the wrong places. Pin them to the originals: EXP is the CARRADA
# experiment (read-only source of src/ and its frozen inputs), ROOT is the project root.
EXP = "{exp}"
ROOT = "{root}"
# ----------------------------------------------------------------------------------------------
'''


def patch(text: str, name: str) -> str:
    # Insert the E6 header right after the EXP/ROOT definitions so later constants can use it.
    anchor = re.search(r"^ROOT = .*$", text, flags=re.M)
    if not anchor:
        raise SystemExit("%s: could not find ROOT definition" % name)
    # Forward slashes: valid on Windows, and immune to escape-level confusion in the emitted
    # source (a raw string plus doubled backslashes would have produced a literal double
    # separator).
    ins = HEADER.format(
        e6=E6.replace("\\", "/"), radius=RADIUS,
        exp=os.path.join(ROOT, "experiments", "r_carrada_real_data_validation").replace("\\", "/"),
        root=ROOT.replace("\\", "/"))
    text = text[:anchor.end()] + "\n" + ins + text[anchor.end():]

    # Override the radius immediately after the detectors import.
    text = re.sub(r"^(import detectors as det.*)$",
                  r"\1\ndet.NMS_SIZE = E6_NMS_RADIUS          # E6 PATCH",
                  text, count=1, flags=re.M)

    # Redirect writes and the threshold handoff into e6; leave other EXP reads alone.
    text = text.replace('os.path.join(EXP, "configs", "tinycnn_zero_shot_thresholds.json")',
                        'os.path.join(E6_DIR, "configs", "tinycnn_zero_shot_thresholds.json")')
    text = text.replace('os.path.join(EXP, "results", "r1_test")',
                        'os.path.join(E6_DIR, "results", "p4_carrada", "r1_test")')
    return text


def main():
    os.makedirs(DST, exist_ok=True)
    for name in SCRIPTS:
        s = open(os.path.join(SRC, name), encoding="utf-8").read()
        out = patch(s, name)
        if "det.NMS_SIZE = E6_NMS_RADIUS" not in out:
            print("  WARNING %s: no detectors import found; radius NOT overridden" % name)
        with open(os.path.join(DST, name), "w", encoding="utf-8") as f:
            f.write(out)
        print("  patched %s  (radius %d)" % (name, RADIUS))
    print("copies in", os.path.relpath(DST, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
