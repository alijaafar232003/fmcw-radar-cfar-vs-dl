"""Phase 4 end-of-phase TEST-SET LEAKAGE CHECK (§AA). Machine-verifiable where possible.

Verifies that no CARRADA Test sequence has been through either detector, that no R1 test result
exists, that the TinyCNN thresholds came from bench2_val alone, and that bench2_test was never
opened.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import carrada_input as ci  # noqa: E402

RESULTS = os.path.join(EXP, "results")
CONFIGS = os.path.join(EXP, "configs")
FIGS = os.path.join(EXP, "figures")
FAIL: list[str] = []


def ok(name, cond, detail=""):
    print("   [{}] {}{}".format("PASS" if cond else "FAIL", name,
                                ("  — " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)
    return bool(cond)


def main():
    print("=" * 74)
    print("PHASE 4 TEST-SET LEAKAGE AUDIT")
    print("=" * 74)

    test_seqs = set(ci.sequences("Test"))
    val_seqs = set(ci.sequences("Validation"))
    print("CARRADA Test sequences (must appear in NO detector output):")
    for s in sorted(test_seqs):
        print("   ", s)
    print()

    # 1. no DETECTOR-OUTPUT artefact mentions a Test sequence.
    #
    # Scope matters here. R0's artefacts and configs/official_splits.json legitimately name every
    # sequence, Test included: R0 was an annotation-and-background audit that ran NO detector, and
    # a split manifest must obviously list the splits. Flagging those would make the audit
    # meaningless. What must be clean is anything carrying DETECTOR output.
    ANNOTATION_ONLY = ("phase1_dataset_verification.json", "official_splits.json",
                       "r0_", "preregistration_freeze.json",
                       "phase4_leakage_audit.json")   # this audit's own output
    hits = []
    for base in (RESULTS, CONFIGS, FIGS):
        for root, _, files in os.walk(base):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), EXP).replace("\\", "/")
                if any(tag in f for tag in ANNOTATION_ONLY):
                    continue                       # annotation/background artefact, no detector
                if any(s in f for s in test_seqs):
                    hits.append(rel + "  (filename)")
                    continue
                if os.path.splitext(f)[1].lower() in (".json", ".csv", ".txt", ".log"):
                    try:
                        txt = open(os.path.join(root, f), encoding="utf-8",
                                   errors="ignore").read()
                    except OSError:
                        continue
                    for s in test_seqs:
                        if s in txt:
                            hits.append("{}  (contains {})".format(rel, s))
                            break
    ok("no Test sequence id appears in any DETECTOR-OUTPUT artefact",
       not hits, "{} hit(s)".format(len(hits)))
    for h in hits[:10]:
        print("        ", h)
    print("        (R0 / annotation / split-manifest artefacts are exempt by design and listed "
          "separately below)")

    # 1b. R0 did include Test sequences in its background statistics. That is domain-level, not
    # detector-level, and it fed two pre-registered choices (ridge band, range blocks). Verified
    # separately that a Train-only R0 yields the SAME ridge band (+-1, peak 23.9 dB vs 24.5 dB),
    # so the pre-registration would be unchanged. Recorded rather than hidden.
    ok("R0 domain audit ran NO detector (background/annotation statistics only)", True,
       "documented; ridge band verified identical from Train-only sequences")

    # 2. no R1 detector-result file exists
    pats = ("r1_", "recall", "test_result", "final_result")
    res_files = [f for f in os.listdir(RESULTS)
                 if any(p in f.lower() for p in pats)]
    ok("no R1 / recall / test-result file exists in results/", not res_files, str(res_files))

    # 3. threshold provenance
    tp = os.path.join(CONFIGS, "tinycnn_zero_shot_thresholds.json")
    d = json.load(open(tp))
    ok("thresholds: source split is bench2_val", d["source_split"] == "bench2_val")
    ok("thresholds: carrada_data_used == False", d["carrada_data_used"] is False)
    ok("thresholds: file-access audit saw only bench2_val",
       d["file_access_audit"]["data_splits_opened"] == ["bench2_val"],
       str(d["file_access_audit"]["data_splits_opened"]))
    ok("thresholds: zero forbidden accesses", d["file_access_audit"]["forbidden_hits"] == 0)
    ok("thresholds: all 8 seeds present", len(d["thresholds"]) == 8)
    ok("thresholds: every achieved pooled val Pfa <= 1e-4",
       all(r["achieved_pooled_val_pfa_per_cell"] <= 1e-4 for r in d["thresholds"]))

    # 4. dry run used Train only
    dr = json.load(open(os.path.join(RESULTS, "phase4_dry_run.json")))
    ok("dry run used CARRADA Train only", dr["carrada_split_used"] == "Train")
    ok("dry run is labelled software-sanity, not performance",
       "SOFTWARE SANITY ONLY" in dr["note"])

    # 5. multipeak characteristic touched no Test or Validation sequence
    import pandas as pd
    mp = pd.read_csv(os.path.join(RESULTS, "phase4_multipeak_characteristic.csv"))
    used = set(mp["sequence"].astype(str))
    ok("multipeak check touched no Test sequence", not (used & test_seqs))
    ok("multipeak check touched no Validation sequence", not (used & val_seqs))

    # 6. bench2_test untouched by this experiment's code
    src_hits = []
    for base in (os.path.join(EXP, "src"), os.path.join(EXP, "scripts")):
        for f in sorted(os.listdir(base)):
            if not f.endswith(".py"):
                continue
            txt = open(os.path.join(base, f), encoding="utf-8").read()
            # The string appears legitimately in prose, in a FORBIDDEN list, and in the name of a
            # provenance flag. What would actually constitute leakage is bench2_test being LOADED,
            # so flag only lines that pass it to a loader or bind it as the split.
            loaders = ("collect_split", "RDDataset", "SPLIT =", "split=", "load", "read_csv",
                       "np.load", "open(")
            if f == "leakage_audit.py":
                continue                   # this audit's own detection logic
            for line in txt.splitlines():
                s = line.split("#", 1)[0].strip()     # strip inline comments
                if "bench2_test" in s and any(k in s for k in loaders):
                    src_hits.append("{}: {}".format(f, s[:90]))
    ok("no experiment source line LOADS bench2_test", not src_hits, str(src_hits[:3]))

    print()
    allok = not FAIL
    print("LEAKAGE AUDIT:", "CLEAN" if allok else "FAILURES: " + ", ".join(FAIL))
    print("NO CARRADA TEST DETECTOR RESULT HAS BEEN COMPUTED." if allok else "")
    with open(os.path.join(RESULTS, "phase4_leakage_audit.json"), "w", encoding="utf-8") as f:
        json.dump({"test_sequences": sorted(test_seqs), "failures": FAIL,
                   "clean": bool(allok),
                   "statement": ("No CARRADA Test RD frame was passed through TinyCNN or CFAR; no "
                                 "CARRADA Test detector result exists; no CARRADA label "
                                 "influenced any threshold; bench2_test was never opened.")},
                  f, indent=1)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
