#!/usr/bin/env python3
"""Run sim2sim on every clip of a library and aggregate Tier-B metrics.

Usage:
    python tools/eval_unified.py --library whole_body_tracking/motions/f1_unified_library.npz \
        --checkpoint trained_models/unified_model_7999.pt --out_dir eval/unified_v1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--clips", type=str, nargs="*", default=None)
    ap.add_argument("--max_steps", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lib = np.load(args.library, allow_pickle=True)
    names = [str(x) for x in lib["clip_names"]]
    skills = [int(x) for x in lib["skill_ids"]]
    if args.clips:
        names = [n for n in names if n in set(args.clips)]
    rows = []
    for name in names:
        out_json = args.out_dir / f"{name}.json"
        cmd = [PY, "whole_body_tracking/scripts/play_unified.py",
               "--library", str(args.library), "--checkpoint", str(args.checkpoint),
               "--clip", name, "--headless", "--json_out", str(out_json),
               "--metrics", str(args.out_dir / f"{name}.npz")]
        if args.max_steps:
            cmd += ["--max_steps", str(args.max_steps)]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if out_json.exists():
            d = json.load(open(out_json))
            d["skill"] = skills[names.index(name)]
            rows.append(d)
            print(f"[{name}] success={d['success']} entry={d['entry_ok']} exit={d['exit_ok']} "
                  f"falls={d['falls']} body_err={d['body_pos_err_rel_mean']}")
        else:
            print(f"[{name}] FAILED\n{r.stdout[-500:]}\n{r.stderr[-800:]}")

    def agg(skill: int):
        sub = [r for r in rows if r["skill"] == skill]
        if not sub:
            return {}
        return {
            "n": len(sub),
            "success_rate": round(float(np.mean([r["success"] for r in sub])) * 100, 1),
            "entry_rate": round(float(np.mean([r["entry_ok"] for r in sub])) * 100, 1),
            "exit_rate": round(float(np.mean([r["exit_ok"] for r in sub])) * 100, 1),
            "fall_rate": round(float(np.mean([r["falls"] for r in sub])) * 100, 1),
            "body_pos_err_mean": round(float(np.mean([r["body_pos_err_rel_mean"] for r in sub])), 4),
            "joint_pos_err_mean": round(float(np.mean([r["joint_pos_err_mean"] for r in sub])), 4),
            "contact_mismatch_pct": round(float(np.mean([r["contact_mismatch_pct"] for r in sub])), 3),
            "slip_mean_mps": round(float(np.mean([r["slip_mean_mps"] for r in sub])), 4),
            "flight_ref_pct": round(float(np.mean([r["flight_ref_pct"] for r in sub])), 2),
            "flight_robot_pct": round(float(np.mean([r["flight_robot_pct"] for r in sub])), 2),
            "action_rate_mean": round(float(np.mean([r["action_rate_mean"] for r in sub])), 4),
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "library": str(args.library),
        "n_clips": len(rows),
        "per_skill": {"idle": agg(0), "walk": agg(1), "run": agg(2), "dance": agg(3)},
        "clips": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["per_skill"], indent=2))


if __name__ == "__main__":
    main()
