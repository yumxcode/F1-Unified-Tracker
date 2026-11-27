"""Build a time-scaled (slower) copy of a motion library.

The user relaxed the speed requirement but requires stable walking, running
and dancing.  Slowing the reference down makes tracking easier and removes
the dynamic phase that causes falls (e.g. dance2_subject1 mid-motion tilt).

Usage:
    python tools/build_slow_library.py \
        --src whole_body_tracking/motions/f1_unified_library.npz \
        --dst whole_body_tracking/motions/f1_unified_library_slow07.npz \
        --scale 0.7

Velocities are scaled by `scale`; positions are resampled at the new rate.
"""
import argparse
import numpy as np


def resample_clip(arr: np.ndarray, scale: float, velocity: bool = False) -> np.ndarray:
    """Resample a (T, ...) array to T/scale frames with linear interpolation."""
    T = arr.shape[0]
    new_T = int(round((T - 1) / scale)) + 1
    t_new = np.arange(new_T) * scale
    t_new = np.clip(t_new, 0, T - 1)
    idx0 = np.floor(t_new).astype(int)
    idx1 = np.minimum(idx0 + 1, T - 1)
    w = (t_new - idx0).reshape((-1,) + (1,) * (arr.ndim - 1))
    out = arr[idx0] * (1 - w) + arr[idx1] * w
    if velocity:
        out = out * scale
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--scale", type=float, default=0.7)
    ap.add_argument("--suffix", default="_slow07")
    args = ap.parse_args()

    d = np.load(args.src)
    names = [str(x) for x in d["clip_names"]]
    starts = [int(x) for x in d["clip_start"]]
    ends = [int(x) for x in d["clip_end"]]
    skills = [int(x) for x in d["skill_ids"]]

    out = {k: [] for k in [
        "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
        "joint_pos", "joint_vel", "foot_contact", "idle_mask"]}
    new_starts, new_ends, new_names = [], [], []
    cursor = 0
    for i, n in enumerate(names):
        s, e = starts[i], ends[i]
        for k in out:
            arr = d[k][s:e]
            if arr.ndim == 0:
                arr = arr[None]
            vel = k in ("body_lin_vel_w", "body_ang_vel_w", "joint_vel")
            out[k].append(resample_clip(arr, args.scale, velocity=vel))
        new_len = out["body_pos_w"][-1].shape[0]
        new_starts.append(cursor)
        new_ends.append(cursor + new_len)
        new_names.append(n + args.suffix)
        cursor += new_len
        print(f"{n}: {e - s} -> {new_len} frames")

    data = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    data.update({
        "fps": np.array([d["fps"]]),
        "clip_names": np.array(new_names),
        "clip_start": np.array(new_starts, dtype=np.int64),
        "clip_end": np.array(new_ends, dtype=np.int64),
        "skill_ids": np.array(skills, dtype=np.int64),
    })
    # preserve any remaining keys (e.g. joint_names) unchanged
    for k in d.files:
        if k not in data:
            data[k] = d[k]
    np.savez_compressed(args.dst, **data)
    total = sum(e - s for s, e in zip(new_starts, new_ends))
    print(f"wrote {args.dst}: {len(new_names)} clips, {total} frames")


if __name__ == "__main__":
    main()
