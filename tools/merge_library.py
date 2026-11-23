"""Merge frame-contiguous library parts produced by split_library.py."""
import argparse
import glob

import numpy as np


def merge(parts, out_path):
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
    np.savez_compressed(out_path, **out)
    print(f"[merge] {out_path}: {len(out['clip_names'])} clips, {len(out['body_pos_w'])} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="glob, e.g. 'motions/f1_unified_library_part*.npz'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    parts = sorted(glob.glob(args.parts))
    assert parts, args.parts
    merge(parts, args.out)


if __name__ == "__main__":
    main()
