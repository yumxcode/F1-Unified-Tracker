#!/usr/bin/env python3
"""Batch curate LAFAN1 -> F1 clips: retarget, contact-repair, gate, select window.

For each clip in the manifest:
  1. GMR retarget the full BVH (cached).
  2. Contact-repair the full CSV with the source BVH contact schedule (cached).
  3. Run the per-gait 9-gate checker with --scan_windows; take the longest
     clean window (>= min_window frames).
  4. Slice the repaired CSV to that window and save a gate report for the slice.

Outputs:
  data/curated/<clip>.csv           selected clean window (36 cols)
  data/curated/<clip>.json          gate report for the selected window
  data/curated/curation_report.json summary
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
BVH_DIR = Path("/Users/yumx/code/Beyondminic-Weilai-X1/downloads/lafan1")
CACHE = REPO / ".tmp/curate"
OUT = REPO / "data/curated"

CLIPS = [
    # name, gait, arm_relock, stance_widen, min_window (frames @30fps)
    ("walk1_subject1", "walk", "c8.0_r0.85", 0.075, 600),
    ("walk1_subject2", "walk", "c8.0_r0.85", 0.075, 600),
    ("walk1_subject5", "walk", "c8.0_r0.85", 0.075, 600),
    ("walk2_subject1", "walk", "c8.0_r0.85", 0.075, 600),
    ("walk3_subject1", "walk", "c8.0_r0.85", 0.075, 600),
    ("walk4_subject1", "walk", "c8.0_r0.85", 0.075, 600),
    ("run1_subject2", "run", "c8.0_r0.85", 0.050, 600),
    ("run1_subject5", "run", "c8.0_r0.85", 0.050, 600),
    ("run2_subject1", "run", "c8.0_r0.85", 0.050, 600),
    ("run2_subject4", "run", "c8.0_r0.85", 0.050, 600),
    ("dance1_subject1", "dance", "c3.0_r0.92", 0.075, 450),
    ("dance1_subject2", "dance", "c3.0_r0.92", 0.075, 450),
    ("dance1_subject3", "dance", "c3.0_r0.92", 0.075, 450),
    ("dance2_subject1", "dance", "c3.0_r0.92", 0.075, 450),
]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("[run]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, gait, arm, widen, min_win in CLIPS:
        bvh = BVH_DIR / f"{name}.bvh"
        raw_csv = CACHE / f"{name}_raw.csv"
        rep_csv = CACHE / f"{name}_rep.csv"
        rep_json = CACHE / f"{name}_rep.json"
        entry = {"gait": gait, "bvh": str(bvh)}

        if not raw_csv.exists():
            r = run([PY, "tools_bvh_to_csv_f1.py", "--bvh_file", str(bvh),
                     "--output_file", str(raw_csv), "--solver", "quadprog",
                     "--arm_relock", arm, "--stance_widen", str(widen),
                     "--clamp_margin", "0.06"])
            if r.returncode != 0:
                entry["error"] = "retarget failed: " + r.stderr[-500:]
                report[name] = entry
                print(entry["error"])
                continue
        if not rep_csv.exists():
            rep_args = [PY, "tools/repair/contact_repair.py", "--csv", str(raw_csv),
                        "--out_csv", str(rep_csv), "--bvh", str(bvh),
                        "--iters", "40", "--vel_scale", "1.0",
                        "--flat_cost", "1000", "--foot_cost", "1000",
                        "--base_cost", "1.0", "--ground_rounds", "4",
                        "--velfactor", "0.02", "--no_post_lift"]
            if gait == "dance":
                rep_args += ["--stance_widen_repair", "0.03"]
            r = run(rep_args)
            if r.returncode != 0:
                entry["error"] = "repair failed: " + r.stderr[-500:]
                report[name] = entry
                print(entry["error"])
                continue
        # scan the repaired full clip for the longest clean window
        r = run([PY, "tools_check_motion_f1.py", "--csv", str(rep_csv),
                 "--gait", gait, "--scan_windows", "--min_window", str(min_win),
                 "--bvh", str(bvh), "--bvh_offset", "0",
                 "--json_out", str(rep_json)])
        m = re.search(r"longest clean window: \[(\d+), (\d+)\) len=(\d+)", r.stdout)
        if not m:
            entry["error"] = "no clean window"
            entry["stdout_tail"] = r.stdout[-1000:]
            report[name] = entry
            print(f"[{name}] no clean window")
            continue
        a, b, ln = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if ln < min_win:
            entry["error"] = f"clean window {ln} < {min_win}"
            report[name] = entry
            print(f"[{name}] window too short {ln}")
            continue
        arr = np.loadtxt(rep_csv, delimiter=",")
        out_csv = OUT / f"{name}_{gait}.csv"
        np.savetxt(out_csv, arr[a:b], delimiter=",", fmt="%.8f")
        # gate report for the selected slice
        slice_json = OUT / f"{name}_{gait}.json"
        r2 = run([PY, "tools_check_motion_f1.py", "--csv", str(out_csv),
                  "--gait", gait, "--bvh", str(bvh), "--bvh_offset", str(a),
                  "--json_out", str(slice_json)])
        passed = False
        if slice_json.exists():
            d = json.load(open(slice_json))
            passed = bool(d.get("overall_pass", False))
        entry.update({"window": [a, b], "frames": ln, "seconds": round(ln / 30.0, 2),
                      "csv": str(out_csv), "pass": passed})
        report[name] = entry
        print(f"[{name}] {gait} window {a}:{b} ({ln}f, {ln/30:.1f}s) pass={passed}")

    (OUT / "curation_report.json").write_text(json.dumps(report, indent=2))
    ok = sum(1 for v in report.values() if v.get("pass"))
    print(f"[done] {ok}/{len(CLIPS)} clips passed")


if __name__ == "__main__":
    main()
