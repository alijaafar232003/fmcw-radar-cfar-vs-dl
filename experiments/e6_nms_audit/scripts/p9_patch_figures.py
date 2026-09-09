"""Phase 6d - isolated copies of the two CARRADA figure scripts, pointed at the revised outputs.

Same discipline as `p4_patch_carrada.py`: the originals in
experiments/r_carrada_real_data_validation/scripts/ are never edited. These copies differ only in
where they READ results from and WRITE figures to.

  make_r1_figures.py        reads results/r1_test/  ->  e6 results/p4_carrada/r1_test/
                            writes figures/r1_test/ ->  e6 figures/r1_test/
  make_crossdomain_figure.py  the same, plus the synthetic half of the figure switches from the
                            radius-5 frozen bootstrap to the radius-2 final bootstrap.

`EXP` itself is left alone, because both scripts import the CARRADA adapters from EXP/src and those
are shared, read-only code.
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
SCRIPTS = ("make_r1_figures.py", "make_crossdomain_figure.py", "make_rd_examples_figure.py")

# Scripts that run a detector live (rather than reading preserved peaks) need the radius override
# and the radius-2 thresholds, exactly as `p4_patch_carrada.py` applies them to the run scripts.
LIVE = ("make_rd_examples_figure.py",)


def patch(text: str, live: bool = False) -> str:
    e6 = E6.replace("\\", "/")
    anchor = re.search(r"^ROOT = .*$", text, flags=re.M)
    # EXP is pinned to the ORIGINAL CARRADA experiment: both scripts import the shared adapters
    # from EXP/src, and these copies live somewhere else, so the __file__ derivation would miss
    # them. Only the result and figure directories are redirected.
    header = ('\n# --------------------------------------------------- E6 PATCH (auto-generated)\n'
              '# Isolated copy; the original is untouched. Only output/result paths change.\n'
              'E6_DIR = "%s"\n'
              'EXP = "%s"\n'
              'ROOT = "%s"\n'
              '# ------------------------------------------------------------------------------\n'
              % (e6, os.path.join(ROOT, "experiments",
                                  "r_carrada_real_data_validation").replace("\\", "/"),
                 ROOT.replace("\\", "/")))
    text = text[:anchor.end()] + "\n" + header + text[anchor.end():]

    text = text.replace('os.path.join(EXP, "results", "r1_test")',
                        'os.path.join(E6_DIR, "results", "p4_carrada", "r1_test")')
    # The cross-domain script joins the filename in the same call, so the pattern above misses it.
    text = text.replace('os.path.join(EXP, "results", "r1_test", "r1_summary.json")',
                        'os.path.join(E6_DIR, "results", "p4_carrada", "r1_test",\n'
                        '                                 "r1_summary.json")')
    text = text.replace('os.path.join(EXP, "figures", "r1_test")',
                        'os.path.join(E6_DIR, "figures", "r1_test")')
    # The per-sequence title was hard-coded to "ahead in all 6", which the revised protocol makes
    # false. Derive it from the data instead, so it cannot go stale again.
    text = text.replace(
        'ax.set_title("Per test sequence — CA-CFAR ahead in all 6", fontsize=9.5)',
        '_nlead = sum(1 for _r in R["per_sequence"] if _r["delta"] < 0)          # E6 PATCH\n'
        '    ax.set_title("Per test sequence — CA-CFAR ahead in %d of %d"\n'
        '                 % (_nlead, len(R["per_sequence"])), fontsize=9.5)')
    # With the revised protocol both bars reach ~0.87, so an in-axes legend sits on top of one of
    # them. Move it below the axes.
    text = text.replace('ax.legend(fontsize=7)\n',
                        'ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06),\n'
                        '              ncol=3, frameon=False)\n', 1)
    # Range blocks were ordered by dict insertion, which put "110-160" before "60-110". Sort them
    # numerically. Also zoom the strata axis: every value sits above 0.8 and a 0-1 axis hides the
    # differences the panel exists to show. The n= annotations move with it.
    text = text.replace(
        '        ks = list(d)\n',
        '        ks = sorted(d, key=lambda s: int(s.split("-")[0])\n'
        '                    if s.split("-")[0].isdigit() else 0)          # E6 PATCH\n')
    text = text.replace(
        '            ax.annotate("n={}".format(d[k]["eligible_objects"]), (i, 0.03), ha="center",',
        '            ax.annotate("n={}".format(d[k]["eligible_objects"]), (i, 0.715), ha="center",')
    text = text.replace('        ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=9.5)',
                        '        ax.set_ylim(0.70, 1.02); ax.set_title(title, fontsize=9.5)')
    # Markdown emphasis does not render in a matplotlib title; it printed literal asterisks.
    text = text.replace('"Unmatched-detection burden\\n**not** a false-alarm rate"',
                        '"Unmatched-detection burden\\nthis is NOT a false-alarm rate"')
    # The tolerance ladder crosses over under the revised protocol, and a 0-1 axis hides it.
    text = text.replace('ax.set_ylim(0, 1.02); ax.set_ylabel("recall"); ax.legend(fontsize=7.5)',
                        'ax.set_ylim(0.80, 1.0); ax.set_ylabel("recall")          # E6 PATCH\n'
                        '    ax.legend(fontsize=7.5, loc="lower right")')
    # The cross-domain figure's synthetic row must come from the SAME protocol as its real row.
    text = text.replace('os.path.join(ROOT, "results", "dl",\n'
                        '                                   "bootstrap_frozen_ep021_test_tol1.csv")',
                        'os.path.join(E6_DIR, "results", "final",\n'
                        '                                   "bootstrap_final_r%d_tol1.csv")' % RADIUS)
    # ---------------------------------------------------------------- cross-domain framing
    # Several titles and the summary box were written for the superseded result, which the
    # revised protocol makes false: the difference no longer reverses, CA-CFAR no longer leads
    # every sequence, and the synthetic split is no longer bench2_test. Derive the counts from the
    # data and restate the framing.
    text = text.replace(
        'axB.set_title("B · Synthetic difference — TinyCNN ahead in 6 of 7", fontsize=10,'
        ' loc="left",',
        'axB.set_title("B · Synthetic difference — TinyCNN ahead in %d of %d"          # E6 PATCH\n'
        '                  % (int((syn["diff"] > 0).sum()), len(syn)), fontsize=10, loc="left",')
    text = text.replace(
        'axD.set_title("D · Real difference — CA-CFAR ahead in 6 of 6", fontsize=10, loc="left",',
        'axD.set_title("D · Real difference — CA-CFAR ahead in %d of %d"               # E6 PATCH\n'
        '                  % (sum(1 for _r in R1["per_sequence"] if _r["delta"] < 0),\n'
        '                     len(R1["per_sequence"])), fontsize=10, loc="left",')
    # Panel C's in-bar labels were placed just below the top of the old, shorter bars; the revised
    # bars are taller, so the rotated text ran off the axis. Anchor them lower.
    text = text.replace('axC.annotate("TinyCNN", (-0.2, cats[0][2] - 0.045)',
                        'axC.annotate("TinyCNN", (-0.2, cats[0][2] - 0.30)')
    text = text.replace('axC.annotate("CA-CFAR", (0.2, cats[0][3] - 0.05)',
                        'axC.annotate("CA-CFAR", (0.2, cats[0][3] - 0.30)')
    # The footer box sat on top of panel D's x-axis label.
    text = text.replace('fig.text(0.5, 0.016,', 'fig.text(0.5, -0.012,')
    # Panel D's axis limits and the "CFAR better" note were placed for the superseded deltas,
    # which were four times larger. Scale both to the data.
    text = text.replace(
        '    axD.set_xlim(-0.30, 0.06)\n'
        '    axD.annotate("← CFAR better", xy=(-0.22, 6.9), fontsize=7.5, color=C_CF,'
        ' ha="center")',
        '    _lim = max(0.05, 1.35 * max(abs(min(dd + [BS["ci_lo"]])),                # E6 PATCH\n'
        '                                abs(max(dd + [BS["ci_hi"]]))))\n'
        '    axD.set_xlim(-_lim, _lim)\n'
        '    axD.annotate("← CFAR better", xy=(-0.62 * _lim, 6.9), fontsize=7.5, color=C_CF,\n'
        '                 ha="center")\n'
        '    axD.annotate("TinyCNN better →", xy=(0.62 * _lim, 6.9), fontsize=7.5, color=C_DL,\n'
        '                 ha="center")')
    text = text.replace(
        '"The advantage reverses. Left column: levels within each domain.  "',
        '"The synthetic advantage does not carry across. Left column: levels within each '
        'domain.  "')
    text = text.replace(
        '"bench2_test, 8 seeds, epoch 21  ·  frozen benchmark, read not recomputed",',
        '"fresh held-out split, 8 seeds, epoch 21  ·  read once, not recomputed",')
    text = text.replace(
        '"The comparable statement is the RANKING: TinyCNN leads in the matched synthetic "\n'
        '             "domain (B), CA-CFAR leads in the unseen real domain (D). Vertical scales '
        'differ "\n             "between the right-hand panels; both are anchored at zero.",',
        '"The comparable statement is the WITHIN-DOMAIN DIFFERENCE: TinyCNN leads throughout "\n'
        '             "the matched synthetic domain (B), while in the unseen real domain (D) the '
        'pooled\\n"\n'
        '             "difference is small and its interval includes zero. Vertical scales differ '
        '"\n             "between the right-hand panels; both are anchored at zero.",')
    # `z["peaks"]` inside a dict comprehension decompresses and re-allocates the ENTIRE peaks
    # array once per frame -- roughly two thousand full copies. At the superseded radius that was
    # merely wasteful; the revised radius leaves four times as many candidates, and the same loop
    # drove the working set past 20 GB and into thrashing. Hoisting the read out of the
    # comprehension changes nothing about the result.
    text = text.replace(
        '    z = np.load(os.path.join(OUT, "{}_peaks.npz".format(tag)), allow_pickle=True)\n'
        '    return {(str(a), str(b)): z["peaks"][int(c):int(d)] for a, b, c, d in z["index"]}',
        '    z = np.load(os.path.join(OUT, "{}_peaks.npz".format(tag)), allow_pickle=True)\n'
        '    pk, ix = z["peaks"], z["index"]          # E6 PATCH: read each array ONCE\n'
        '    return {(str(a), str(b)): pk[int(c):int(d)] for a, b, c, d in ix}')

    if live:
        # This script runs the detectors itself, so it needs the radius override and the
        # radius-2 thresholds, not the preserved radius-5 outputs.
        text = re.sub(r"^(import detectors as det.*)$",
                      r"\1\ndet.NMS_SIZE = %d          # E6 PATCH" % RADIUS,
                      text, count=1, flags=re.M)
        text = text.replace(
            'os.path.join(EXP, "configs", "tinycnn_zero_shot_thresholds.json")',
            'os.path.join(E6_DIR, "configs", "tinycnn_zero_shot_thresholds.json")')
    return text


def main():
    os.makedirs(DST, exist_ok=True)
    for name in SCRIPTS:
        s = open(os.path.join(SRC, name), encoding="utf-8").read()
        out = patch(s, live=name in LIVE)
        if "E6_DIR" not in out:
            raise SystemExit("%s: patch did not apply" % name)
        with open(os.path.join(DST, name), "w", encoding="utf-8") as f:
            f.write(out)
        print("  patched", name)
    print("copies in", os.path.relpath(DST, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
