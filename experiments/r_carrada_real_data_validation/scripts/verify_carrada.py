"""Phase 1 (CARRADA real-data external validation): verify the cleaned CARRADA dataset and resolve annotation terminology.

STRICTLY READ-ONLY with respect to CARRADA. Writes only into the E5 results/ folder.

Covers Phase 1 items 5 (integrity), 6 (terminology), 7 (official splits).
Runs NO detector and computes NO detection statistic.
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

CARRADA = r"C:/Users/USER/Desktop/CARRADA/Carrada"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
META = os.path.join(ROOT, "data", "carrada_meta")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

CLASS_NAMES = {1: "pedestrian", 2: "cyclist", 3: "car"}


def npy_header(path):
    """Read shape/dtype from an .npy header without loading the array."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version[0] == 1:
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    return shape, str(dtype), fortran


def main():
    report = {}

    # ---------- load metadata (from the project-local backup, itself hash-verified) ----------
    light = json.load(open(os.path.join(META, "light_dataset_frame_oriented.json")))
    selected = json.load(open(os.path.join(META, "selected_light_dataset_frame_oriented.json")))
    fo = json.load(open(os.path.join(META, "annotations_frame_oriented.json")))
    seq_ref = json.load(open(os.path.join(META, "data_seq_ref.json")))
    validated = [l.strip() for l in open(os.path.join(META, "validated_seqs.txt")) if l.strip()]

    seqs = sorted(d for d in os.listdir(CARRADA)
                  if os.path.isdir(os.path.join(CARRADA, d)) and d[0].isdigit())

    print("=" * 70)
    print("1-3. SEQUENCES AND FRAME COUNTS")
    print("=" * 70)
    n_numpy = n_raw = n_cam = 0
    per_seq = {}
    for s in seqs:
        dn = os.path.join(CARRADA, s, "range_doppler_numpy")
        dr = os.path.join(CARRADA, s, "range_doppler_raw")
        dc = os.path.join(CARRADA, s, "camera_images")
        fn = sorted(f for f in os.listdir(dn)) if os.path.isdir(dn) else []
        fr = sorted(f for f in os.listdir(dr)) if os.path.isdir(dr) else []
        fc = sorted(f for f in os.listdir(dc)) if os.path.isdir(dc) else []
        per_seq[s] = {"numpy": len(fn), "raw": len(fr), "camera": len(fc),
                      "aligned": fn == fr}
        n_numpy += len(fn); n_raw += len(fr); n_cam += len(fc)
    print("sequences on disk          : {}".format(len(seqs)))
    print("sequences in validated_seqs: {}".format(len(validated)))
    print("sequences in data_seq_ref  : {}".format(len(seq_ref)))
    print("_numpy RD frames           : {}".format(n_numpy))
    print("_raw    RD frames          : {}".format(n_raw))
    print("camera images retained     : {}".format(n_cam))

    print("\n" + "=" * 70)
    print("4. FILENAME ALIGNMENT BETWEEN _numpy AND _raw")
    print("=" * 70)
    misaligned = [s for s, v in per_seq.items() if not v["aligned"]]
    print("sequences with identical filename sets: {}/{}".format(
        len(seqs) - len(misaligned), len(seqs)))
    if misaligned:
        print("  MISALIGNED:", misaligned)
    report["aligned_numpy_raw"] = not misaligned

    print("\n" + "=" * 70)
    print("11. EVERY FRAME IN THE LIGHT LIST PRESENT ON DISK?")
    print("=" * 70)
    missing_n, missing_r = [], []
    for s in sorted(light):
        for fr in light[s]:
            if not os.path.isfile(os.path.join(CARRADA, s, "range_doppler_numpy", fr + ".npy")):
                missing_n.append(s + "/" + fr)
            if not os.path.isfile(os.path.join(CARRADA, s, "range_doppler_raw", fr + ".npy")):
                missing_r.append(s + "/" + fr)
    exp = sum(len(v) for v in light.values())
    print("frames expected from light list : {}".format(exp))
    print("missing from _numpy             : {}".format(len(missing_n)))
    print("missing from _raw               : {}".format(len(missing_r)))

    print("\n" + "=" * 70)
    print("12-14. SHAPE / DTYPE / READABILITY  (header scan of ALL RD files)")
    print("=" * 70)
    shapes = collections.Counter(); dtypes = collections.Counter()
    bad_header, checked = [], 0
    for s in seqs:
        for sub in ("range_doppler_numpy", "range_doppler_raw"):
            d = os.path.join(CARRADA, s, sub)
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                try:
                    shape, dt, _ = npy_header(p)
                    shapes[(sub, shape)] += 1; dtypes[(sub, dt)] += 1
                except Exception as exc:
                    bad_header.append((p, repr(exc)))
                checked += 1
    print("headers scanned      : {}".format(checked))
    for k, v in sorted(shapes.items()):
        print("  shape {:24s} {:6d}".format(str(k), v))
    for k, v in sorted(dtypes.items()):
        print("  dtype {:24s} {:6d}".format(str(k), v))
    print("unreadable headers   : {}".format(len(bad_header)))
    for p, e in bad_header[:5]:
        print("   BAD", p, e)

    # full-load spot check for genuine corruption (headers can be fine while data is truncated)
    rng = np.random.default_rng(0)
    sample, bad_load = [], []
    for s in seqs:
        fr = sorted(os.listdir(os.path.join(CARRADA, s, "range_doppler_numpy")))
        for f in rng.choice(fr, size=min(10, len(fr)), replace=False):
            sample.append((s, str(f)))
    for s, f in sample:
        for sub in ("range_doppler_numpy", "range_doppler_raw"):
            p = os.path.join(CARRADA, s, sub, f)
            try:
                a = np.load(p)
                if a.shape != (256, 64) or not np.isfinite(a).all():
                    bad_load.append((p, "shape {} finite {}".format(a.shape, np.isfinite(a).all())))
            except Exception as exc:
                bad_load.append((p, repr(exc)))
    print("full-load spot checks: {} files, {} failures".format(len(sample) * 2, len(bad_load)))
    for p, e in bad_load[:5]:
        print("   BAD", p, e)

    print("\n" + "=" * 70)
    print("6-10. METADATA PRESENCE")
    print("=" * 70)
    need_meta = ["annotations_frame_oriented.json", "annotations_instance_oriented.json",
                 "data_seq_ref.json", "selected_light_dataset_frame_oriented.json",
                 "light_dataset_frame_oriented.json", "validated_seqs.txt",
                 "rd_stats.json", "rd_stats_all.json", "rd_weights.json",
                 "instance_exceptions.json", "LICENSE", "README.md"]
    for f in need_meta:
        a = os.path.isfile(os.path.join(CARRADA, f))
        b = os.path.isfile(os.path.join(META, f))
        print("  {:48s} source {:5s}  backup {:5s}".format(f, str(a), str(b)))
    n_box = sum(os.path.isfile(os.path.join(CARRADA, s, "annotations", "box",
                                            "range_doppler_light.json")) for s in seqs)
    print("  annotations/box range_doppler_light.json    : {}/{} sequences".format(n_box, len(seqs)))
    print("  metadata backup data/carrada_meta exists    : {}".format(os.path.isdir(META)))

    print("\n" + "=" * 70)
    print("6. ANNOTATION TERMINOLOGY  (this table defines the R1 denominators)")
    print("=" * 70)
    total_frames = sum(len(v) for v in light.values())
    annotated = sum(1 for s in fo for f in fo[s] if len(fo[s][f]) > 0)
    empty = sum(1 for s in fo for f in fo[s] if len(fo[s][f]) == 0)
    obj_frames = sum(len(fo[s][f]) for s in fo for f in fo[s])
    tracks_fo = {s: {ik for f in fo[s] for ik in fo[s][f]} for s in fo}
    n_tracks_fo = sum(len(v) for v in tracks_fo.values())
    n_tracks_ref = sum(len(v["instances"]) for v in seq_ref.values())
    sel_frames = sum(len(v) for v in selected.values())
    cls_objframes = collections.Counter()
    cls_tracks = collections.Counter()
    for s in fo:
        seen = {}
        for f in fo[s]:
            for ik, v in fo[s][f].items():
                lab = v["range_doppler"]["label"]
                cls_objframes[lab] += 1
                seen[ik] = lab
        for ik, lab in seen.items():
            cls_tracks[lab] += 1
    hist = collections.Counter(len(fo[s][f]) for s in fo for f in fo[s])

    rows = [
        ("total frames (light list)", total_frames,
         "every frame with an RD array on disk"),
        ("annotated frames", annotated,
         "frames with >=1 object; == selected_light list ({})".format(sel_frames)),
        ("empty frames", empty, "frames with 0 objects; usable as false-alarm-only"),
        ("frame-level object instances", obj_frames,
         "sum over frames of objects present  <-- the 8,750"),
        ("unique tracks (frame-oriented)", n_tracks_fo,
         "distinct instance ids across all frames"),
        ("unique tracks (data_seq_ref)", n_tracks_ref,
         "instances listed in data_seq_ref.json  <-- the 84"),
    ]
    print("{:34s} {:>8s}   {}".format("quantity", "count", "meaning"))
    for name, val, meaning in rows:
        print("{:34s} {:8d}   {}".format(name, val, meaning))
    print("\nobjects-per-frame histogram: {}".format(dict(sorted(hist.items()))))
    print("\n{:14s} {:>16s} {:>14s}".format("class", "object-frames", "unique tracks"))
    for lab in sorted(CLASS_NAMES):
        print("{:14s} {:16d} {:14d}".format(CLASS_NAMES[lab], cls_objframes[lab], cls_tracks[lab]))
    print("{:14s} {:16d} {:14d}".format("TOTAL", sum(cls_objframes.values()),
                                        sum(cls_tracks.values())))

    consistent = (annotated + empty == total_frames) and (annotated == sel_frames)
    print("\nconsistency: annotated + empty == total frames  -> {}".format(
        annotated + empty == total_frames))
    print("consistency: annotated == selected_light count -> {}".format(annotated == sel_frames))

    # ---- reconcile the 84-vs-77 track discrepancy explicitly ----
    exceptions = json.load(open(os.path.join(META, "instance_exceptions.json")))
    listed_raw = sum(len(v["instances"]) for v in seq_ref.values())
    dup_per_seq = {}
    for s, v in seq_ref.items():
        c = collections.Counter(v["instances"])
        d = {k: n for k, n in c.items() if n > 1}
        if d:
            dup_per_seq[s] = d
    n_dup = sum(n - 1 for d in dup_per_seq.values() for n in d.values())
    listed_unique = sum(len(set(v["instances"])) for v in seq_ref.values())
    n_merged = sum(len(v) for v in exceptions.values())
    io_path = os.path.join(META, "annotations_instance_oriented.json")
    io = json.load(open(io_path))
    n_tracks_io = sum(len(io[s]) for s in io)

    print("\nreconciliation of the track count:")
    print("  {:58s} {:4d}".format("raw sum of data_seq_ref 'instances' list lengths", listed_raw))
    print("  {:58s} {:4d}   {}".format("minus duplicate entries in those lists", -n_dup,
                                       dup_per_seq))
    print("  {:58s} {:4d}".format("= unique listed instance ids", listed_unique))
    print("  {:58s} {:4d}   (instance_exceptions.json)".format(
        "minus instances merged into another during annotation", -n_merged))
    print("  {:58s} {:4d}".format("= unique annotated tracks", listed_unique - n_merged))
    print("  {:58s} {:4d}".format("independently, tracks in annotations_frame_oriented",
                                  n_tracks_fo))
    print("  {:58s} {:4d}".format("independently, tracks in annotations_instance_oriented",
                                  n_tracks_io))
    tracks_reconciled = (listed_unique - n_merged) == n_tracks_fo == n_tracks_io
    print("  reconciled: {}".format(tracks_reconciled))
    consistent = consistent and tracks_reconciled
    report["track_reconciliation"] = {
        "data_seq_ref_raw_sum": listed_raw,
        "duplicate_entries": dup_per_seq,
        "duplicates_removed": n_dup,
        "unique_listed_ids": listed_unique,
        "merged_by_instance_exceptions": n_merged,
        "merge_map": exceptions,
        "unique_annotated_tracks": listed_unique - n_merged,
        "tracks_frame_oriented": n_tracks_fo,
        "tracks_instance_oriented": n_tracks_io,
        "reconciled": bool(tracks_reconciled),
    }

    print("\n" + "=" * 70)
    print("7. OFFICIAL SEQUENCE-LEVEL SPLITS  (from data_seq_ref.json)")
    print("=" * 70)
    splits = collections.defaultdict(list)
    for s, v in seq_ref.items():
        splits[v["split"]].append(s)
    split_out = {}
    print("{:12s} {:>5s} {:>9s} {:>11s} {:>9s} {:>9s} {:>8s}".format(
        "split", "seqs", "frames", "annotated", "empty", "obj-frms", "tracks"))
    for sp in ("Train", "Validation", "Test"):
        ss = sorted(splits[sp])
        fr = sum(len(light[s]) for s in ss)
        an = sum(1 for s in ss for f in fo[s] if len(fo[s][f]) > 0)
        em = sum(1 for s in ss for f in fo[s] if len(fo[s][f]) == 0)
        ob = sum(len(fo[s][f]) for s in ss for f in fo[s])
        tr = sum(len(tracks_fo[s]) for s in ss)
        split_out[sp] = {"sequences": ss, "n_sequences": len(ss), "frames": fr,
                         "annotated_frames": an, "empty_frames": em,
                         "object_frames": ob, "unique_tracks": tr}
        print("{:12s} {:5d} {:9d} {:11d} {:9d} {:9d} {:8d}".format(
            sp, len(ss), fr, an, em, ob, tr))
    tot_seq = sum(v["n_sequences"] for v in split_out.values())
    print("{:12s} {:5d} {:9d} {:11d} {:9d} {:9d} {:8d}".format(
        "TOTAL", tot_seq,
        sum(v["frames"] for v in split_out.values()),
        sum(v["annotated_frames"] for v in split_out.values()),
        sum(v["empty_frames"] for v in split_out.values()),
        sum(v["object_frames"] for v in split_out.values()),
        sum(v["unique_tracks"] for v in split_out.values())))
    print("\nevery sequence assigned exactly one split : {}".format(tot_seq == len(seqs)))
    print("split sets disjoint                      : {}".format(
        len(set().union(*[set(v["sequences"]) for v in split_out.values()])) == tot_seq))

    # ---------- persist ----------
    os.makedirs(OUT, exist_ok=True)
    report.update({
        "carrada_root": CARRADA,
        "n_sequences_on_disk": len(seqs),
        "n_sequences_validated_txt": len(validated),
        "n_sequences_data_seq_ref": len(seq_ref),
        "n_rd_numpy_frames": n_numpy,
        "n_rd_raw_frames": n_raw,
        "n_camera_images": n_cam,
        "filename_alignment_numpy_raw": not misaligned,
        "missing_numpy": missing_n,
        "missing_raw": missing_r,
        "npy_headers_scanned": checked,
        "npy_shapes": {str(k): v for k, v in shapes.items()},
        "npy_dtypes": {str(k): v for k, v in dtypes.items()},
        "unreadable_headers": bad_header,
        "full_load_spot_checks": len(sample) * 2,
        "full_load_failures": bad_load,
        "box_annotation_sequences": n_box,
        "terminology": {name: val for name, val, _ in rows},
        "objects_per_frame_histogram": {str(k): v for k, v in sorted(hist.items())},
        "class_object_frames": {CLASS_NAMES[k]: v for k, v in cls_objframes.items()},
        "class_unique_tracks": {CLASS_NAMES[k]: v for k, v in cls_tracks.items()},
        "counts_consistent": bool(consistent),
        "official_splits": split_out,
    })
    with open(os.path.join(OUT, "phase1_dataset_verification.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "official_splits.json"), "w", encoding="utf-8") as f:
        json.dump({"source": "CARRADA data_seq_ref.json (official, sequence-level)",
                   "policy": "sequence-level only; never split by frame",
                   "splits": {k: v["sequences"] for k, v in split_out.items()}}, f, indent=1)

    ok = (not misaligned and not missing_n and not missing_r and not bad_header
          and not bad_load and consistent and tot_seq == len(seqs))
    print("\n" + "=" * 70)
    print("PHASE 1 DATASET VERIFICATION: {}".format("PASS" if ok else "ISSUES FOUND"))
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
