"""CARRADA input loading and dense-annotation reconstruction. Experiment-local, READ-ONLY.

Nothing here writes to the CARRADA tree. Any derived cache belongs under the experiment folder.

Representation (fixed by R0, PREREGISTRATION §2):
    range_doppler_numpy is power in dB up to an unknown constant.
      CFAR    : P = 10**((D - c)/10) for any constant c   (ratio test, c cancels)
      TinyCNN : mag = 10**(D/20), fed to the FROZEN normalize_rd, which then computes
                10*log10(mag**2 / median(mag**2)) == D - median(D) exactly.
"""
from __future__ import annotations

import json
import os

import numpy as np

CARRADA_ROOT = r"C:/Users/USER/Desktop/CARRADA/Carrada"
_HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(_HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
META = os.path.join(ROOT, "data", "carrada_meta")

RD_SHAPE = (256, 64)
CLASS_NAMES = {1: "pedestrian", 2: "cyclist", 3: "car"}

_CACHE: dict = {}


# ------------------------------------------------------------------ metadata
def metadata():
    """Load the (hash-verified) project-local metadata backup once."""
    if "meta" not in _CACHE:
        _CACHE["meta"] = {
            "light": json.load(open(os.path.join(META, "light_dataset_frame_oriented.json"))),
            "fo": json.load(open(os.path.join(META, "annotations_frame_oriented.json"))),
            "seq_ref": json.load(open(os.path.join(META, "data_seq_ref.json"))),
        }
    return _CACHE["meta"]


def sequences(split: str) -> list[str]:
    """Official CARRADA sequence-level split. `split` in {Train, Validation, Test}."""
    ref = metadata()["seq_ref"]
    return sorted(s for s, v in ref.items() if v["split"] == split)


def frames(seq: str) -> list[str]:
    return sorted(metadata()["light"][seq])


# ------------------------------------------------------------------ RD frames
def load_rd_db(seq: str, frame: str) -> np.ndarray:
    """One CARRADA RD frame as power in dB. Verified, never modified."""
    p = "{}/{}/range_doppler_numpy/{}.npy".format(CARRADA_ROOT, seq, frame)
    d = np.load(p)
    if d.shape != RD_SHAPE:
        raise ValueError("{}: shape {} != {}".format(p, d.shape, RD_SHAPE))
    if d.dtype != np.float64:
        raise ValueError("{}: dtype {} != float64".format(p, d.dtype))
    if not np.isfinite(d).all():
        raise ValueError("{}: non-finite values".format(p))
    return d


def to_cfar_power(db: np.ndarray, c: float | None = None) -> np.ndarray:
    """Linear power for CFAR. `c` is one constant per frame; CFAR is a ratio test so it cancels.

    Default c = frame median, which centres the exponent and keeps the dynamic range small.
    """
    if c is None:
        c = float(np.median(db))
    return np.power(10.0, (db - c) / 10.0)


def to_tinycnn_magnitude(db: np.ndarray) -> np.ndarray:
    """Magnitude for the FROZEN TinyCNN preprocessing.

    The frozen `src/dl/dataset.normalize_rd` takes magnitude, squares it to power, divides by the
    frame median and takes 10*log10. Feeding mag = 10**(D/20) makes that pipeline compute
    D - median(D) exactly, so the registered adapter is realised by the frozen code itself rather
    than by a reimplementation. `validate_adapters.py` verifies the identity numerically.
    """
    return np.power(10.0, db / 20.0)


def adapter_reference(db: np.ndarray) -> np.ndarray:
    """The registered adapter written out explicitly, as the verification reference only.

        relative_db = D - median(D); clip(-10, +60); / 20
    """
    x = db - float(np.median(db))
    return np.clip(x, -10.0, 60.0) / 20.0


# ------------------------------------------------------------------ annotations
def objects(seq: str, frame: str) -> list[dict]:
    """Every annotated object in one frame, with its dense support reconstructed from JSON.

    The bulk .npy annotation folders were deleted after Phase 1 proved them byte-identical to this
    JSON on 360/360 sampled frame-kind pairs. No interpolation is performed and no coordinate is
    invented: the cells are exactly CARRADA's own.
    """
    fo = metadata()["fo"]
    out = []
    for inst_id, v in fo.get(seq, {}).get(frame, {}).items():
        rd = v["range_doppler"]
        dense = np.asarray(rd["dense"], dtype=int).reshape(-1, 2)
        sparse = np.asarray(rd["sparse"], dtype=int).reshape(-1, 2)
        out.append({
            "sequence": seq, "frame": frame, "object_id": inst_id,
            "label": rd["label"], "class": CLASS_NAMES.get(rd["label"], "?"),
            "dense": dense, "sparse": sparse, "box": rd["box"],
        })
    return out


def dense_mask(cells: np.ndarray, shape=RD_SHAPE) -> np.ndarray:
    """Boolean support map from CARRADA dense cells. Out-of-bounds cells would be a data error."""
    m = np.zeros(shape, dtype=bool)
    if len(cells) == 0:
        return m
    r, d = cells[:, 0], cells[:, 1]
    if r.min() < 0 or r.max() >= shape[0] or d.min() < 0 or d.max() >= shape[1]:
        raise ValueError("annotation cell outside {}".format(shape))
    m[r, d] = True
    return m
