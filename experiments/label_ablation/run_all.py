"""Launch all (arm, seed) training runs with a concurrency limit. Writes only under this folder.

24 runs = 3 arms x 8 seeds, 21 epochs each. At ~580 s/epoch that is ~3.4 h per run and ~81 CPU-hours
in total, so they are run concurrently: each process is pinned to `threads=2` (the frozen,
load-bearing value), and several processes run at once.

Concurrency does NOT affect results. Torch's CPU reduction order depends on the thread count WITHIN
a process, which is fixed at 2 regardless of how many processes are running.

Already-finished runs are skipped, so this is safe to re-run after an interruption.

    python experiments/label_ablation/run_all.py --workers 6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "venv" / "Scripts" / "python.exe"
ARMS = ("point", "flat3x3", "gaussian")
SEEDS = range(8)
FINAL_EPOCH = 21


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=21)
    args = ap.parse_args()

    (HERE / "logs").mkdir(parents=True, exist_ok=True)
    (HERE / "checkpoints").mkdir(parents=True, exist_ok=True)

    jobs = []
    for arm in ARMS:
        for s in SEEDS:
            done = HERE / "checkpoints" / f"tinycnn_{arm}_s{s}_ep{FINAL_EPOCH:03d}.pt"
            (print(f"skip {arm}_s{s} (already finished)") if done.exists()
             else jobs.append((arm, s)))

    print(f"{len(jobs)} runs to do, {args.workers} at a time, threads={args.threads} each")
    if not jobs:
        return

    t0 = time.perf_counter()
    running, queue, done_n = [], list(jobs), 0

    while queue or running:
        while queue and len(running) < args.workers:
            arm, s = queue.pop(0)
            out = (HERE / "logs" / f"stdout_{arm}_s{s}.log").open("w", encoding="utf-8")
            p = subprocess.Popen(
                [str(PY), str(HERE / "run_training.py"), "--arm", arm, "--seed", str(s),
                 "--epochs", str(args.epochs), "--threads", str(args.threads)],
                stdout=out, stderr=subprocess.STDOUT, cwd=str(ROOT))
            running.append((p, arm, s, out))
            print(f"  started {arm}_s{s}  ({len(running)} running, {len(queue)} queued)",
                  flush=True)

        time.sleep(20)

        for entry in list(running):
            p, arm, s, out = entry
            if p.poll() is None:
                continue
            out.close()
            running.remove(entry)
            done_n += 1
            el = (time.perf_counter() - t0) / 3600
            status = "OK" if p.returncode == 0 else f"FAILED rc={p.returncode}"
            print(f"  finished {arm}_s{s}: {status}  "
                  f"[{done_n}/{len(jobs)} done, {el:.2f} h elapsed]", flush=True)

    print(f"\nall runs finished in {(time.perf_counter() - t0) / 3600:.2f} h")
    n = len(list((HERE / 'checkpoints').glob('tinycnn_*_ep*.pt')))
    print(f"checkpoints on disk: {n} (expect {len(ARMS) * len(SEEDS) * 4})")


if __name__ == "__main__":
    main()
