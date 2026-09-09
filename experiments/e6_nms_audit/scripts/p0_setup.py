"""Phase 0 - create the isolated structure and write MANIFEST.json.

Records the git commit, the frozen selection-rule hash, the frozen checkpoint hashes, the
environment, and the CARRADA isolation status at start. Writes only inside the experiment folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(EXP))

DIRS = [
    "configs", "scripts", "logs", "data",
    "results/p1_selection", "results/p3_synth", "results/p4_carrada", "results/p5_audit",
]


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    for d in DIRS:
        os.makedirs(os.path.join(EXP, d), exist_ok=True)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()

    ckpts = {}
    for s in range(8):
        p = os.path.join(ROOT, "results", "dl", "tinycnn_frozen_s%d_ep021.pt" % s)
        ckpts["tinycnn_frozen_s%d_ep021.pt" % s] = sha256(p) if os.path.exists(p) else None

    inputs = {}
    for rel in ("configs/cfar_alpha_table.json", "configs/frozen_dl.yaml",
                "configs/bench2_test.yaml", "src/cfar/detect.py", "src/cfar/cfar.py",
                "src/dl/infer.py", "src/dl/models.py", "src/eval/detection.py",
                "src/radar/config.py"):
        p = os.path.join(ROOT, rel)
        inputs[rel] = sha256(p) if os.path.exists(p) else None

    man = {
        "experiment": "e6_nms_audit",
        "purpose": "Fast audit: does a smaller NMS radius materially change the main "
                   "synthetic and CARRADA conclusions?",
        "status": "PHASE_0_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_start": commit,
        "selection_rule": {
            "path": "SELECTION_RULE.md",
            "sha256": sha256(os.path.join(EXP, "SELECTION_RULE.md")),
            "frozen_before_any_measurement": True,
        },
        "frozen_checkpoints": ckpts,
        "reused_inputs_sha256": inputs,
        "environment": {
            "python": platform.python_version(),
            "platform": sys.platform,
        },
        "carrada_isolation_at_start": {
            "verified": True,
            "frozen_files_changed": 2,
            "violations": ["docs/dl_design.md", "docs/literature_corpus_review.md"],
            "violation_category": "project_doc",
            "experimental_categories_changed": 0,
            "assessment": (
                "PRE-EXISTING AND DECLARED. Both files are documentation, corrected earlier in "
                "the reporting work at the user's explicit request (dl_design.md: the stale "
                "441-cell figure and the 'same information' claim; literature_corpus_review.md: "
                "the Liu/Banerjee attribution). Neither is read by this experiment. Every "
                "experimental category - cfar_source, dl_source, eval_source, radar_source, "
                "frozen_dl_checkpoint, headline_result, experiment_e2/e3/e4_output, "
                "cfar_baseline_result, data_source, dataset_config, cfar_alpha_table, "
                "preregistration - re-hashed unchanged. The CARRADA manifest was NOT "
                "re-baselined: an old artefact is never rewritten because a new experiment "
                "disagrees with it."
            ),
        },
        "commands": [],
        "artefacts": {},
    }

    out = os.path.join(EXP, "MANIFEST.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)

    print("created:", ", ".join(DIRS))
    print("git commit :", commit)
    print("rule sha256:", man["selection_rule"]["sha256"][:16], "...")
    print("checkpoints hashed:", sum(1 for v in ckpts.values() if v))
    print("wrote", os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
