"""A process-wide filesystem allow-list for CARRADA access. Phase 1.

The data rule for this experiment is absolute: fine-tuning and every development decision may see
ONLY the official CARRADA TRAIN sequences. CARRADA VALIDATION is the final untouched evaluation
and may not be opened until the freeze; CARRADA TEST may not be opened at all.

A comment saying so is not enforcement. This installs a `sys.addaudithook` that inspects every
file `open` in the process and raises `PermissionError` the moment a path under the CARRADA root
resolves to a sequence outside the allowed set. It cannot be defeated by an indirect path, a glob,
a helper module or a library call, because it sits below all of them at the interpreter's own
audit hook.

    from guard import install, ALLOW_TRAIN, access_report
    install(ALLOW_TRAIN)          # fine-tuning, thresholds, internal development
    install(ALLOW_VALIDATION)     # ONLY the final frozen evaluation script

Both modes always forbid Test. `access_report()` returns the sequences actually touched, which is
written into every artefact so the claim is checkable after the fact rather than asserted.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
R_EXP = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")

CARRADA_ROOT = os.path.normcase(os.path.normpath(r"C:/Users/USER/Desktop/CARRADA/Carrada"))
_SPLITS = json.load(open(os.path.join(R_EXP, "configs", "official_splits.json"),
                         encoding="utf-8"))["splits"]

TRAIN = tuple(_SPLITS["Train"])
VALIDATION = tuple(_SPLITS["Validation"])
TEST = tuple(_SPLITS["Test"])

ALLOW_TRAIN = "train"
ALLOW_VALIDATION = "validation"

_state = {"mode": None, "allowed": frozenset(), "opened": set(), "installed": False}


def _sequence_of(path: str):
    """The CARRADA sequence a path belongs to, or None if the path is not under CARRADA."""
    try:
        p = os.path.normcase(os.path.normpath(os.path.abspath(path)))
    except Exception:
        return None
    if not p.startswith(CARRADA_ROOT + os.sep):
        return None
    rest = p[len(CARRADA_ROOT) + 1:].split(os.sep)
    return rest[0] if rest and rest[0] else None


def _hook(event, args):
    if event != "open" or not args:
        return
    target = args[0]
    if not isinstance(target, (str, bytes, os.PathLike)):
        return
    path = os.fspath(target)
    if isinstance(path, bytes):
        path = path.decode("utf-8", "replace")
    seq = _sequence_of(path)
    if seq is None:
        return
    if seq not in _state["allowed"]:
        where = ("TEST" if seq in TEST else
                 "VALIDATION" if seq in VALIDATION else
                 "TRAIN" if seq in TRAIN else "UNKNOWN")
        raise PermissionError(
            "BLOCKED CARRADA ACCESS: sequence %r belongs to the %s split, which this process is "
            "not permitted to open (mode=%r). Path: %s" % (seq, where, _state["mode"], path))
    _state["opened"].add(seq)


def install(mode):
    """Install the allow-list. Idempotent per mode; refuses to widen an already-installed mode."""
    if mode == ALLOW_TRAIN:
        allowed = frozenset(TRAIN)
    elif mode == ALLOW_VALIDATION:
        # The final evaluation scores every arm on Validation. Train stays readable because the
        # frozen thresholds and the internal-dev artefacts are read from disk, not recomputed.
        allowed = frozenset(VALIDATION) | frozenset(TRAIN)
    else:
        raise ValueError("unknown mode: %r" % (mode,))

    if _state["installed"]:
        if _state["mode"] != mode:
            raise RuntimeError("guard already installed as %r; refusing to switch to %r"
                               % (_state["mode"], mode))
        return
    assert not (allowed & frozenset(TEST)), "TEST must never be allowed"
    _state.update(mode=mode, allowed=allowed, installed=True)
    sys.addaudithook(_hook)
    print("[guard] CARRADA access restricted to %s (%d sequences); TEST is blocked"
          % (mode, len(allowed)), flush=True)


def access_report():
    op = sorted(_state["opened"])
    return {
        "mode": _state["mode"],
        "n_allowed": len(_state["allowed"]),
        "sequences_opened": op,
        "n_opened": len(op),
        "opened_train": sorted(set(op) & set(TRAIN)),
        "opened_validation": sorted(set(op) & set(VALIDATION)),
        "opened_test": sorted(set(op) & set(TEST)),
        "test_opened": bool(set(op) & set(TEST)),
        "validation_opened": bool(set(op) & set(VALIDATION)),
    }


def selftest():
    """Prove the hook actually fires, rather than trusting that it would."""
    install(ALLOW_TRAIN)
    ok_train = os.path.join(CARRADA_ROOT, TRAIN[0], "range_doppler_numpy")
    assert os.path.isdir(ok_train), "expected the train sequence to exist"

    for split_name, seq in (("TEST", TEST[0]), ("VALIDATION", VALIDATION[0])):
        p = os.path.join(CARRADA_ROOT, seq, "range_doppler_numpy")
        f = sorted(os.listdir(p))[0]
        try:
            open(os.path.join(p, f), "rb").close()
        except PermissionError as e:
            print("  %-11s blocked as expected: %s" % (split_name, str(e)[:72] + "..."))
        else:
            raise AssertionError("%s was NOT blocked -- the guard is not working" % split_name)

    p = os.path.join(CARRADA_ROOT, TRAIN[0], "range_doppler_numpy")
    f = sorted(os.listdir(p))[0]
    open(os.path.join(p, f), "rb").close()
    print("  TRAIN       allowed as expected")
    print("  access report:", json.dumps(access_report()["opened_train"][:1]))
    return True


if __name__ == "__main__":
    selftest()
    print("guard selftest PASSED")
