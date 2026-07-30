import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from radar.config import RadarConfig, derive_radar_params


def save_labels_csv(output_dir: Path, labels: list[dict]) -> Path:
    labels_path = output_dir / "labels.csv"

    fieldnames = [
        "target_id",
        "range_m",
        "velocity_mps",
        "amplitude_db",
        "range_bin",
        "velocity_bin",
        "nearest_range_m",
        "nearest_velocity_mps",
        "cell_snr_db",
    ]

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for target_id, item in enumerate(labels):
            writer.writerow(
                {
                    "target_id": target_id,
                    "range_m": item["range_m"],
                    "velocity_mps": item["velocity_mps"],
                    "amplitude_db": item["amplitude_db"],
                    "range_bin": item["range_bin"],
                    "velocity_bin": item["velocity_bin"],
                    "nearest_range_m": item["nearest_range_m"],
                    "nearest_velocity_mps": item["nearest_velocity_mps"],
                    "cell_snr_db": item.get("cell_snr_db", float("nan")),
                }
            )

    return labels_path


def save_frame(
    output_dir: Path,
    rd_db: np.ndarray,
    rd_mag: np.ndarray,
    label_mask: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    labels: list[dict],
    cfg: RadarConfig,
    snr_db: float,
    seed: int | None,
    scenario: str,
    clutter: dict,
    multipath: dict,
    condition: str | None = None,
    noise_var: float | None = None,
    snr_processing_gain_db: float | None = None,
    compact: bool = False,
) -> dict:
    """Write one frame.

    compact=True omits per-frame files that are redundant for large datasets:
      * rd_map_db.npy   — a fixed transform of rd_map_mag (20*log10, peak-normalised)
      * range/velocity axes — identical in every frame; stored once per dataset instead.
    This cuts frame size by roughly 60% with no loss of information.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "rd_map_mag.npy", rd_mag.astype(np.float32))
    np.save(output_dir / "label_mask.npy", label_mask.astype(np.uint8))

    if not compact:
        np.save(output_dir / "rd_map_db.npy", rd_db.astype(np.float32))
        np.save(output_dir / "range_axis.npy", range_axis.astype(np.float32))
        np.save(output_dir / "velocity_axis.npy", velocity_axis.astype(np.float32))

    save_labels_csv(output_dir, labels)

    p = derive_radar_params(cfg)

    metadata = {
        "scenario": scenario,
        # Stratification key: which designed condition this frame belongs to (stratified
        # mixture datasets only). Enables per-condition evaluation with guaranteed sample counts.
        "condition": condition,
        # Requested cell SNR for a 0 dB-amplitude reference target. Each target's own
        # measured cell SNR is in its label entry (and in labels.csv).
        "reference_cell_snr_db": float(snr_db),
        "snr_db": float(snr_db),
        "noise_var": float(noise_var) if noise_var is not None else None,
        "snr_processing_gain_db": (
            float(snr_processing_gain_db) if snr_processing_gain_db is not None else None
        ),
        "seed": int(seed) if seed is not None else None,
        "radar_config": asdict(cfg),
        "derived": asdict(p),
        "rd_shape": list(rd_db.shape),
        "num_targets": len(labels),
        "targets": labels,
        "clutter": clutter,
        "multipath": multipath,
        "compact": bool(compact),
        "files": (
            {
                "rd_map_mag": "rd_map_mag.npy",
                "label_mask": "label_mask.npy",
                "labels_csv": "labels.csv",
                "metadata": "metadata.json",
                "axes": "../../axes.npz (shared by all frames)",
            }
            if compact
            else {
                "rd_map_db": "rd_map_db.npy",
                "rd_map_mag": "rd_map_mag.npy",
                "label_mask": "label_mask.npy",
                "range_axis": "range_axis.npy",
                "velocity_axis": "velocity_axis.npy",
                "labels_csv": "labels.csv",
                "metadata": "metadata.json",
            }
        ),
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata