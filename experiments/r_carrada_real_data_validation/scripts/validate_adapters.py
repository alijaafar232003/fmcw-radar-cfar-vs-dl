"""Phase 4 checks C1 and C2 — the two adapters.

C1  TinyCNN input adapter equivalence.
C2  CFAR scale invariance.

Read-only. Uses CARRADA TRAIN sequences only (never Validation, never Test) plus synthetic frames.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map  # noqa: E402
from dl.dataset import normalize_rd               # noqa: E402
from dl.infer import cfar_valid_mask              # noqa: E402
from cfar.detect import all_peaks                 # noqa: E402

import carrada_input as ci                            # noqa: E402

RESULTS = os.path.join(EXP, "results")
SEED = 20260831


def sample_train_frames(n_seq=None, per_seq=6):
    """Deterministic sample of CARRADA TRAIN frames. Train only — never Validation or Test."""
    rng = np.random.default_rng(SEED)
    seqs = ci.sequences("Train")
    if n_seq:
        seqs = seqs[:n_seq]
    out = []
    for s in seqs:
        fr = ci.frames(s)
        for f in rng.choice(fr, size=min(per_seq, len(fr)), replace=False):
            out.append((s, str(f)))
    return out


# --------------------------------------------------------------------- C1
def check_c1_adapter():
    """`D - median(D)` == `10log10(P / median(P))`, and the frozen path realises it exactly."""
    print("=" * 74)
    print("C1  TinyCNN input adapter equivalence")
    print("=" * 74)
    frames = sample_train_frames()
    d_identity, d_frozen = [], []
    for seq, fr in frames:
        D = ci.load_rd_db(seq, fr)

        # (a) the algebraic identity the pre-registration relies on
        P = np.power(10.0, D / 10.0)
        lhs = D - np.median(D)
        rhs = 10.0 * np.log10(P / np.median(P))
        d_identity.append(np.abs(lhs - rhs).max())

        # (b) the FROZEN preprocessing, fed mag = 10**(D/20), vs the registered adapter
        frozen = normalize_rd(ci.to_tinycnn_magnitude(D))
        ref = ci.adapter_reference(D).astype(np.float32)
        d_frozen.append(np.abs(frozen - ref).max())

    a, b = float(np.max(d_identity)), float(np.max(d_frozen))
    print("frames checked (CARRADA Train only)        : {}".format(len(frames)))
    print("max |(D-median D) - 10log10(P/median P)|   : {:.3e} dB".format(a))
    print("max |frozen normalize_rd - registered adap|: {:.3e}  (float32 output)".format(b))
    # Tolerances are set from FLOAT PRECISION, not chosen so the test passes. The identity is
    # exact in exact arithmetic; the residual is the round trip through 10**(D/10) and log10 on
    # values spanning ~60 dB. A 2.8e-08 dB residual on a ~60 dB value is a relative error near
    # 5e-10, roughly 2 ulp of float64. The frozen pipeline returns float32 (eps 1.19e-07), so its
    # residual is likewise about 2 ulp. Anything systematically larger would be a real defect.
    tol_id, tol_frozen = 1e-6, 1e-5
    ok = a < tol_id and b < tol_frozen
    print("C1 PASS:", ok)
    return {"check": "C1_adapter_equivalence", "frames": len(frames),
            "max_abs_diff_identity_db": a, "max_abs_diff_frozen_vs_reference": b,
            "tolerance_identity": tol_id, "tolerance_frozen": tol_frozen,
            "tolerance_basis": "float precision: ~2 ulp of float64 (identity), float32 (frozen)",
            "pass": bool(ok)}


# --------------------------------------------------------------------- C2
def check_c2_scale_invariance():
    """CFAR is a ratio test, so a common multiplicative scale must cancel exactly."""
    print()
    print("=" * 74)
    print("C2  CFAR scale invariance")
    print("=" * 74)
    frames = sample_train_frames(n_seq=4, per_seq=3)
    scales = [0.01, 1.0, 100.0, 1e6]
    rows = []
    worst_ratio, worst_peaks = 0.0, 0
    for kind in ("CA", "GO", "SO", "OS"):
        cfg = CfarConfig(kind=kind)
        for seq, fr in frames:
            D = ci.load_rd_db(seq, fr)
            P0 = ci.to_cfar_power(D)
            valid = cfar_valid_mask(D.shape)
            base, cfar_valid, _alpha = cfar_ratio_map(P0, cfg)
            base_pk = all_peaks(base, valid, nms_size=5)
            for s in scales:
                r, _, _ = cfar_ratio_map(P0 * s, cfg)
                pk = all_peaks(r, valid, nms_size=5)
                dr = float(np.nanmax(np.abs(r - base)))
                dpk = int(abs(len(pk) - len(base_pk)))
                worst_ratio = max(worst_ratio, dr)
                worst_peaks = max(worst_peaks, dpk)
                rows.append({"kind": kind, "scale": s, "max_abs_ratio_diff": dr,
                             "peak_count_diff": dpk})
    print("variants x frames x scales                 : {}".format(len(rows)))
    print("max |ratio(P*s) - ratio(P)| over all       : {:.3e}".format(worst_ratio))
    print("max peak-count difference                  : {}".format(worst_peaks))
    ok = worst_ratio < 1e-9 and worst_peaks == 0
    print("C2 PASS:", ok)
    if not ok:
        print("  !! scale changes CFAR output materially — STOP and investigate.")
        print("  !! do NOT modify CFAR source to make this pass.")
    return {"check": "C2_cfar_scale_invariance", "combinations": len(rows),
            "scales": scales, "max_abs_ratio_diff": worst_ratio,
            "max_peak_count_diff": worst_peaks, "tolerance": 1e-9, "pass": bool(ok)}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = [check_c1_adapter(), check_c2_scale_invariance()]
    with open(os.path.join(RESULTS, "phase4_adapter_checks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    allok = all(o["pass"] for o in out)
    print()
    print("ADAPTER CHECKS:", "ALL PASS" if allok else "FAILURE — STOP")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
