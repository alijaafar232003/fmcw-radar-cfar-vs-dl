"""Launch all 16 E4 patch runs (masked + unmasked, seeds 0-7) with a concurrency limit."""
from __future__ import annotations
import argparse, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PY = ROOT / "venv" / "Scripts" / "python.exe"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=21)
    a = ap.parse_args()

    jobs = [(arm, s) for arm in ("unmasked", "masked") for s in range(8)
            if not (HERE / "checkpoints" / f"tinycnn_{arm}_s{s}_ep{a.epochs:03d}.pt").exists()]
    print(f"{len(jobs)} runs, {a.workers} at a time", flush=True)

    t0, running, queue, done = time.perf_counter(), [], list(jobs), 0
    while queue or running:
        while queue and len(running) < a.workers:
            arm, s = queue.pop(0)
            f = (HERE / "logs" / f"stdout_{arm}_s{s}.log").open("w", encoding="utf-8")
            p = subprocess.Popen([str(PY), str(HERE / "scripts" / "train_patch.py"),
                                  "--arm", arm, "--seed", str(s), "--epochs", str(a.epochs)],
                                 stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
            running.append((p, arm, s, f))
            print(f"  started {arm}_s{s}", flush=True)
        time.sleep(20)
        for e in list(running):
            p, arm, s, f = e
            if p.poll() is None:
                continue
            f.close(); running.remove(e); done += 1
            print(f"  finished {arm}_s{s}: rc={p.returncode}  "
                  f"[{done}/{len(jobs)}, {(time.perf_counter()-t0)/3600:.2f} h]", flush=True)
    print(f"\nall finished in {(time.perf_counter()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
