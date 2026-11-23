"""Split a multi-clip motion library into frame-contiguous parts.

Each part is a valid library (clip_start/clip_end relative to the part) plus a
``_frame_offset`` scalar giving its start frame in the original library.  Used to
upload a >100 MB library through the GitHub Git Data API (blob limit 100 MB).
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--bounds", required=True, help="clip indices, e.g. 0,6,12,17")
    args = ap.parse_args()

    d = np.load(args.library, allow_pickle=True)
    names = [str(x) for x in d["clip_names"]]
    bounds = [int(x) for x in args.bounds.split(",")]
    assert bounds[0] == 0 and bounds[-1] == len(names), (bounds, len(names))
    T = d["body_pos_w"].shape[0]

    frame_keys, clip_keys = set(), set()
    for k in d.files:
        v = d[k]
        if not isinstance(v, np.ndarray) or v.ndim < 1:
            continue
        if v.shape[0] == T:
            frame_keys.add(k)
        elif v.shape[0] == len(names):
            clip_keys.add(k)

    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        f0, f1 = int(d["clip_start"][a]), int(d["clip_end"][b - 1])
        out = {}
        for k in d.files:
            if k in frame_keys:
                out[k] = d[k][f0:f1]
            elif k in clip_keys:
                v = d[k][a:b]
                if k in ("clip_start", "clip_end"):
                    v = v - f0
                out[k] = v
            else:
                out[k] = d[k]
        out["_frame_offset"] = f0
        path = f"{args.prefix}{i}.npz"
        np.savez_compressed(path, **out)
        print(f"[part] {path}: clips {a}:{b} ({b-a}), frames {f1-f0}, offset {f0}")


if __name__ == "__main__":
    main()
