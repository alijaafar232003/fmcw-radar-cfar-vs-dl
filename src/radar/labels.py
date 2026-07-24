from typing import List

import numpy as np

from radar.simulator import Target


def make_target_labels(
    targets: List[Target],
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    cell_snr_db: List[float] | None = None,
) -> list[dict]:
    """Ground-truth label per target.

    `cell_snr_db` is the measured post-integration SNR of each target in its RD cell
    (see radar.calibration). It is the meaningful difficulty measure — a single
    frame-level SNR is ambiguous when a strong target sits next to a weak one.
    """
    labels = []

    for i, target in enumerate(targets):
        r_bin = int(np.argmin(np.abs(range_axis - target.range_m)))
        v_bin = int(np.argmin(np.abs(velocity_axis - target.velocity_mps)))

        labels.append(
            {
                "range_m": float(target.range_m),
                "velocity_mps": float(target.velocity_mps),
                "amplitude_db": float(target.amplitude_db),
                "range_bin": r_bin,
                "velocity_bin": v_bin,
                "nearest_range_m": float(range_axis[r_bin]),
                "nearest_velocity_mps": float(velocity_axis[v_bin]),
                "cell_snr_db": (
                    float(cell_snr_db[i]) if cell_snr_db is not None else float("nan")
                ),
            }
        )

    return labels


def make_label_mask(
    labels: list[dict],
    shape: tuple[int, int],
    radius_r: int = 1,
    radius_d: int = 1,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)

    for item in labels:
        r = item["range_bin"]
        d = item["velocity_bin"]

        r0 = max(0, r - radius_r)
        r1 = min(shape[0], r + radius_r + 1)

        d0 = max(0, d - radius_d)
        d1 = min(shape[1], d + radius_d + 1)

        mask[r0:r1, d0:d1] = 1

    return mask