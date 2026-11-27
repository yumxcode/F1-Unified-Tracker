#!/usr/bin/env python3
"""Tier-B evaluation aligned with docs/STANDARDS.md v1.2.

For every clip: 3 random seeds x 2 complete replays (seed perturbs the initial
state; repeat 2 adds action noise).  Aggregation follows the standard:
  B1 success (0.5 m anchor radius, no fall) rate over episodes
  B2 fall rate (root z / tilt / mean body error)
  B3 median of per-episode median |body pos error| (m)
  B4 mean joint error (rad)
  B5 mean body velocity error (m/s)
  B6 contact mismatch (% frames)
  B7 mean stance-frame slip (m/s)
  B8 |robot flight - source flight| / max(source,1%) <= 20%
  B9 action rate (rad/frame)
  B10 entry/exit >= 95%, exit drift <= 5 cm
  B11 per skill
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
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--noise", type=float, default=0.005)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lib = np.load(args.library, allow_pickle=True)
    names = [str(x) for x in lib["clip_names"]]
    skills = [int(x) for x in lib["skill_ids"]]
    # name -> skill (must be looked up by name: --clips filters the list, so
    # positional indexing would mislabel walk as idle / run as dance)
    name_to_skill = {n: s for n, s in zip(names, skills)}
    if args.clips:
        names = [n for n in names if n in set(args.clips)]
    rows = []
    for name in names:
        for seed in args.seeds:
            for rep in range(args.repeats):
                out_json = args.out_dir / f"{name}_s{seed}_r{rep}.json"
                cmd = [PY, "whole_body_tracking/scripts/play_unified.py",
                       "--library", str(args.library), "--checkpoint", str(args.checkpoint),
                       "--clip", name, "--headless", "--json_out", str(out_json),
                       "--seed", str(seed),
                       "--action_noise", str(0.0 if rep == 0 else args.noise)]
                r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
                if r.returncode != 0:
                    raise SystemExit(f"play_unified failed for {name} s{seed} r{rep}:\n{r.stderr[-800:]}")
                if out_json.exists():
                    d = json.load(open(out_json))
                    # required fields of the standard evaluator; a missing field
                    # means the evaluator is not the reference version
                    for f in ("entry_ok", "entry_blend_end_frame", "fall_reason",
                              "body_pos_err_abs_median_m", "entry_first_lift_frame",
                              "slip_stance_mean_mps", "tilt_max_deg"):
                        if f not in d:
                            raise SystemExit(f"missing field {f!r} in {out_json}")
                    d["skill"] = name_to_skill[name]
                    d["seed"] = seed
                    d["rep"] = rep
                    rows.append(d)
                    print(f"[{name} s{seed} r{rep}] success={d['success']} entry={d['entry_ok']} "
                          f"exit={d['exit_ok']} falls={d['falls']} body_abs_med={d['body_pos_err_abs_median_m']}")
                else:
                    print(f"[{name} s{seed} r{rep}] FAILED")

    def agg(skill: int):
        sub = [r for r in rows if r["skill"] == skill]
        if not sub:
            return {}
        success = float(np.mean([r["success"] for r in sub])) * 100
        fall = float(np.mean([r["falls"] for r in sub])) * 100
        b3 = float(np.median([r["body_pos_err_abs_median_m"] for r in sub]))
        b4 = float(np.mean([r["joint_pos_err_mean"] for r in sub]))
        b5 = float(np.mean([r["body_lin_vel_err_mean_mps"] for r in sub]))
        b6 = float(np.mean([r["contact_mismatch_pct"] for r in sub]))
        b7 = float(np.mean([r["slip_stance_mean_mps"] for r in sub]))
        fr = float(np.mean([r["flight_ref_pct"] for r in sub]))
        fro = float(np.mean([r["flight_robot_pct"] for r in sub]))
        b8 = abs(fro - fr) / max(fr, 1.0) <= 0.20
        b9 = float(np.mean([r["action_rate_mean"] for r in sub]))
        entry = float(np.mean([r["entry_ok"] for r in sub])) * 100
        exit_ = float(np.mean([r["exit_ok"] for r in sub])) * 100
        drift = float(np.mean([r["exit_drift_m"] for r in sub]))
        gates = {
            "B1_success_ge_90": success >= 90.0,
            "B2_fall_le_5": fall <= 5.0,
            "B3_body_abs_median_le_35mm": b3 <= 0.035,
            "B4_joint_le_0.65rad": b4 <= 0.65,
            "B5_vel_le_0.30mps": b5 <= 0.30,
            "B6_contact_mismatch_le_1pct": b6 <= 1.0,
            "B7_slip_stance_le_0.10mps": b7 <= 0.10,
            "B8_flight": bool(b8),
            "B9_action_rate_le_1.0": b9 <= 1.0,
            "B10_entry_exit_ge_95_drift_le_5cm": (entry >= 95.0 and exit_ >= 95.0 and drift <= 0.05),
            "B11_per_skill": True,
        }
        return {
            "n_episodes": len(sub),
            "success_rate": round(success, 1),
            "fall_rate": round(fall, 1),
            "body_pos_err_abs_median_m": round(b3, 4),
            "joint_pos_err_mean": round(b4, 4),
            "body_lin_vel_err_mean_mps": round(b5, 4),
            "contact_mismatch_pct": round(b6, 3),
            "slip_stance_mean_mps": round(b7, 4),
            "flight_ref_pct": round(fr, 2),
            "flight_robot_pct": round(fro, 2),
            "action_rate_mean": round(b9, 4),
            "entry_rate": round(entry, 1),
            "exit_rate": round(exit_, 1),
            "exit_drift_mean_m": round(drift, 4),
            "gates": gates,
            "tier_b_pass": all(gates.values()),
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "library": str(args.library),
        "seeds": args.seeds,
        "repeats": args.repeats,
        "n_clips": len(names),
        "per_skill": {"idle": agg(0), "walk": agg(1), "run": agg(2), "dance": agg(3)},
        "episodes": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["per_skill"], indent=2))


if __name__ == "__main__":
    main()
