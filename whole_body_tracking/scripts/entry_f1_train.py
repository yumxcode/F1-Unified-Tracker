#!/usr/bin/env python3
"""Gradmotion entry point for F1: install package then launch training.

The motion npz is prebuilt locally (tools_csv_to_npz_f1_local.py — the
container image has a numpy 1.x/2.x split that breaks the Isaac-side
converter, same as the X1 flow since v10).

    python scripts/entry_f1_train.py \
        --task=Tracking-Flat-F1-v0 \
        --motion_file motions/dance_f1_train.npz \
        --headless --num_envs 4096 --max_iterations 5000
"""

from __future__ import annotations

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def _merge_motion_parts() -> None:
    """Rebuild motions/f1_unified_library.npz from uploaded frame-contiguous parts."""
    import glob
    import numpy as np

    parts = sorted(glob.glob(os.path.join(REPO_ROOT, "motions", "f1_unified_library_part*.npz")))
    if not parts:
        return
    loaded = [np.load(p, allow_pickle=True) for p in parts]
    n0 = len(loaded[0]["clip_names"])
    out = {}
    for k in loaded[0].files:
        if k == "_frame_offset":
            continue
        v0 = loaded[0][k]
        if isinstance(v0, np.ndarray) and v0.ndim >= 1 and v0.shape[0] == len(loaded[0]["body_pos_w"]):
            out[k] = np.concatenate([p[k] for p in loaded], axis=0)
        elif isinstance(v0, np.ndarray) and v0.ndim >= 1 and v0.shape[0] == n0:
            chunks, off = [], 0
            for p in loaded:
                v = p[k]
                if k in ("clip_start", "clip_end"):
                    v = v + off
                chunks.append(v)
                off += len(p["body_pos_w"])
            out[k] = np.concatenate(chunks, axis=0)
        else:
            out[k] = v0
    full = os.path.join(REPO_ROOT, "motions", "f1_unified_library.npz")
    np.savez_compressed(full, **out)
    print(f"[entry] merged {len(parts)} motion parts -> {full} ({len(out['body_pos_w'])} frames)")


def main() -> None:
    os.chdir(REPO_ROOT)
    _merge_motion_parts()

    ext_path = os.path.join(REPO_ROOT, "source", "whole_body_tracking")
    print(f"[entry] pip install -e {ext_path} --no-deps")
    # --no-deps: the IsaacLab image already provides every runtime dependency;
    # letting pip resolve deps would upgrade numpy/scipy and break isaac-sim.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ext_path, "--no-deps"])

    train_args = [a for a in sys.argv[1:]]
    if not any(a.startswith("--motion_file") for a in train_args):
        raise SystemExit("entry_f1_train.py requires --motion_file (prebuilt npz)")

    train_py = os.path.join(SCRIPT_DIR, "rsl_rl", "train.py")
    print(f"[entry] exec train.py {' '.join(train_args)}")
    os.execv(sys.executable, [sys.executable, train_py] + train_args)


if __name__ == "__main__":
    main()
