#!/usr/bin/env python3
"""Curate the UMR-retargeted ACCAD walk/run clips for F1.

UMR outputs qpos npz at 120 fps (36 columns: root xyz, quat wxyz, 29 joints in
F1_JOINT_NAMES order).  We convert to the 36-column CSV layout (quat xyzw),
apply the same contact repair used for GMR (source contact is not available for
these npz, so the repair uses an ankle-height stance heuristic), then run the
per-gait 9-gate checker.  Only passing clips are kept.

Usage:
    python tools/batch_umr.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
UMR = Path("/Users/yumx/code/UMR_weilai/umr_data")
OUT = REPO / "data/curated_umr"
CACHE = REPO / ".tmp/umr"

CLIPS = [
    ("accad_walk_B1", "walk", UMR / "weichai_f1_29_accad_walking/B1_stand_to_walk_weichai_f1_29.npz"),
    ("accad_walk_B2", "walk", UMR / "weichai_f1_29_accad_walking/B2_walk_to_stand_weichai_f1_29.npz"),
    ("accad_walk_B3", "walk", UMR / "weichai_f1_29_accad_walking/B3_walk1_weichai_f1_29.npz"),
    ("accad_walk_B4", "walk", UMR / "weichai_f1_29_accad_walking/B4_stand_to_walk_back_weichai_f1_29.npz"),
    ("accad_run_C2", "run", UMR / "weichai_f1_29_accad_running/C2_Run_to_stand_f1.npz"),
    ("accad_run_C3", "run", UMR / "weichai_f1_29_accad_running/C3_Run_f1.npz"),
    ("accad_run_C4", "run", UMR / "weichai_f1_29_accad_running/C4_Run_to_walk_f1.npz"),
    ("accad_run_C5", "run", UMR / "weichai_f1_29_accad_running/C5_Walk_to_run_f1.npz"),
]


def run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, gait, npz in CLIPS:
        if not npz.exists():
            report[name] = {"error": f"missing {npz}"}
            continue
        d = np.load(npz, allow_pickle=True)
        q = d["qpos"].astype(np.float64)
        out = q.copy()
        out[:, 3:7] = q[:, [4, 5, 6, 3]]  # wxyz -> xyzw
        csv = CACHE / f"{name}_raw.csv"
        np.savetxt(csv, out, delimiter=",", fmt="%.8f")
        rep = CACHE / f"{name}_rep.csv"
        r = run([PY, "tools/repair/contact_repair.py", "--csv", str(csv),
                 "--out_csv", str(rep), "--fps", "120", "--stance_h", "0.09",
                 "--stance_widen_repair", "0.055", "--iters", "30",
                 "--vel_scale", "1.0", "--flat_cost", "1000",
                 "--foot_cost", "1000", "--base_cost", "1.0", "--ground_rounds", "4"])
        if r.returncode != 0:
            report[name] = {"error": r.stderr[-500:]}
            continue
        j = OUT / f"{name}_{gait}.json"
        r = run([PY, "tools_check_motion_f1.py", "--csv", str(rep), "--fps", "120",
                 "--gait", gait, "--json_out", str(j)])
        passed = False
        if j.exists():
            passed = bool(json.load(open(j)).get("overall_pass", False))
        out_csv = OUT / f"{name}_{gait}.csv"
        if passed:
            import shutil
            shutil.copy(rep, out_csv)
        report[name] = {"gait": gait, "pass": passed, "csv": str(out_csv) if passed else None}
        print(f"[{name}] {gait} pass={passed}")

    (OUT / "umr_report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] {sum(1 for v in report.values() if v.get('pass'))}/{len(CLIPS)} passed")


if __name__ == "__main__":
    main()
