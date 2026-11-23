#!/usr/bin/env python3
"""Concatenate per-clip npz files into one multi-clip motion library.

Usage:
    python tools/build_library.py --clips data/clips/*.npz --out motions/f1_unified_library.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    clips = sorted(args.clips)
    assert clips, "no clips"
    first = np.load(clips[0], allow_pickle=True)
    body_names = [str(x) for x in first["body_names"]]
    joint_names = [str(x) for x in first["joint_names"]]

    joint_pos, joint_vel = [], []
    body_pos, body_quat = [], []
    body_lin, body_ang = [], []
    foot_contact, idle_mask = [], []
    clip_start, clip_end, clip_names, skill_ids = [], [], [], []

    t = 0
    for p in clips:
        d = np.load(p, allow_pickle=True)
        assert [str(x) for x in d["body_names"]] == body_names, f"body order mismatch: {p}"
        assert [str(x) for x in d["joint_names"]] == joint_names, f"joint order mismatch: {p}"
        n = d["joint_pos"].shape[0]
        joint_pos.append(d["joint_pos"])
        joint_vel.append(d["joint_vel"])
        body_pos.append(d["body_pos_w"])
        body_quat.append(d["body_quat_w"])
        body_lin.append(d["body_lin_vel_w"])
        body_ang.append(d["body_ang_vel_w"])
        foot_contact.append(d["foot_contact"])
        idle_mask.append(d["idle_mask"])
        clip_start.append(t)
        clip_end.append(t + n)
        clip_names.append(str(d["clip_name"]))
        skill_ids.append(int(d["skill_id"]))
        print(f"  + {p.name}: {n} frames, skill {int(d['skill_id'])}, idle {int(d['idle_mask'].sum())}")
        t += n

    out = {
        "fps": 50,
        # float16 halves the file size; MotionLoader casts to float32 on load.
        "joint_pos": np.concatenate(joint_pos).astype(np.float16),
        "joint_vel": np.concatenate(joint_vel).astype(np.float16),
        "body_pos_w": np.concatenate(body_pos).astype(np.float16),
        "body_quat_w": np.concatenate(body_quat).astype(np.float16),
        "body_lin_vel_w": np.concatenate(body_lin).astype(np.float16),
        "body_ang_vel_w": np.concatenate(body_ang).astype(np.float16),
        "body_names": np.array(body_names),
        "joint_names": np.array(joint_names),
        "clip_start": np.array(clip_start, dtype=np.int64),
        "clip_end": np.array(clip_end, dtype=np.int64),
        "clip_names": np.array(clip_names),
        "skill_ids": np.array(skill_ids, dtype=np.int64),
        "foot_contact": np.concatenate(foot_contact).astype(np.float32),
        "idle_mask": np.concatenate(idle_mask).astype(np.float32),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    total = t
    print(f"[library] {args.out}: {len(clips)} clips, {total} frames ({total/50:.1f}s @50Hz)")
    print(f"[library] skills: " + ", ".join(f"{n}={s}" for n, s in zip(clip_names, skill_ids)))


if __name__ == "__main__":
    main()
