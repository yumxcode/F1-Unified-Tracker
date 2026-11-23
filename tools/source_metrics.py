#!/usr/bin/env python3
"""Compute source-mocap gait statistics used to freeze the retargeting gates.

For every LAFAN1 clip of a gait class we compute, on the *human* source:
  * double-support (DS) fraction, using the LAFAN1 contact rule
    (per-frame foot speed < 0.02 m/frame)
  * DS gap / hip-width, mean and P5 (hip width = |LeftUpLeg - RightUpLeg|)
  * stance foot pitch (ankle->toe vector), mean and P95
  * maximum foot lift, flight fraction (both feet off)

Outputs data/source_metrics.json.  The gate table in docs/STANDARDS.md is
derived from this file and then frozen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "GMR"))
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402

BVH_DIR = Path("/Users/yumx/code/Beyondminic-Weilai-X1/downloads/lafan1")
CLIPS = {
    "walk": ["walk1_subject1", "walk1_subject2", "walk1_subject5", "walk2_subject1",
             "walk2_subject3", "walk2_subject4", "walk3_subject1", "walk3_subject2",
             "walk3_subject3", "walk3_subject4", "walk3_subject5", "walk4_subject1"],
    "run": ["run1_subject2", "run1_subject5", "run2_subject1", "run2_subject4",
            "sprint1_subject2", "sprint1_subject4"],
    "dance": ["dance1_subject1", "dance1_subject2", "dance1_subject3",
              "dance2_subject1", "dance2_subject2", "dance2_subject3",
              "dance2_subject4", "dance2_subject5"],
}


def stats(bvh: Path):
    frames, _ = load_bvh_file(str(bvh), format="lafan1")
    L = np.array([f["LeftFoot"][0] for f in frames])
    R = np.array([f["RightFoot"][0] for f in frames])
    Lt = np.array([f["LeftToe"][0] for f in frames])
    Rt = np.array([f["RightToe"][0] for f in frames])
    LH = np.array([f["LeftUpLeg"][0] for f in frames])
    RH = np.array([f["RightUpLeg"][0] for f in frames])
    hipw = np.linalg.norm(LH - RH, axis=1)
    gap = np.linalg.norm((L - R)[:, :2], axis=1)
    vL = np.linalg.norm(np.diff(L, axis=0), axis=1)
    vR = np.linalg.norm(np.diff(R, axis=0), axis=1)
    cl = np.r_[vL < 0.02, False] & (L[:, 2] < 0.12)
    cr = np.r_[vR < 0.02, False] & (R[:, 2] < 0.12)
    ds = cl & cr
    ratio = gap / hipw
    lp = np.degrees(np.arctan2(Lt[:, 2] - L[:, 2], np.linalg.norm((Lt - L)[:, :2], axis=1)))
    rp = np.degrees(np.arctan2(Rt[:, 2] - R[:, 2], np.linalg.norm((Rt - R)[:, :2], axis=1)))
    stance_pitch = np.concatenate([lp[cl], rp[cr]])
    return {
        "frames": len(frames),
        "seconds": round(len(frames) / 30.0, 2),
        "ds_frac": round(float(ds.mean()), 4),
        "gap_over_hipw_mean": round(float(ratio[ds].mean()) if ds.any() else float("nan"), 4),
        "gap_over_hipw_p5": round(float(np.percentile(ratio[ds], 5)) if ds.any() else float("nan"), 4),
        "stance_pitch_mean_deg": round(float(stance_pitch.mean()), 2),
        "stance_pitch_p95_deg": round(float(np.percentile(stance_pitch, 95)), 2),
        "flat_pct_10deg": round(float(np.mean(np.abs(stance_pitch) < 10) * 100), 2),
        "max_foot_lift_m": round(float(max(L[:, 2].max(), R[:, 2].max()) - 0.078), 4),
        "flight_frac": round(float(np.mean(~cl & ~cr)), 4),
    }


def main() -> None:
    out = {}
    for gait, names in CLIPS.items():
        rows = {}
        for n in names:
            p = BVH_DIR / f"{n}.bvh"
            if not p.exists():
                print(f"[skip] {p}")
                continue
            rows[n] = stats(p)
            print(gait, n, rows[n])
        # aggregate per-clip P5
        p5s = [v["gap_over_hipw_p5"] for v in rows.values() if np.isfinite(v["gap_over_hipw_p5"])]
        flat = [v["flat_pct_10deg"] for v in rows.values()]
        out[gait] = {
            "clips": rows,
            "n_clips": len(rows),
            "gap_p5_min": round(float(min(p5s)), 4) if p5s else None,
            "gap_p5_p5": round(float(np.percentile(p5s, 5)), 4) if p5s else None,
            "gap_p5_median": round(float(np.median(p5s)), 4) if p5s else None,
            "flat_pct_min": round(float(min(flat)), 2) if flat else None,
            "flat_pct_median": round(float(np.median(flat)), 2) if flat else None,
        }
        print(gait, "aggregate", out[gait])
    path = REPO / "data/source_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
