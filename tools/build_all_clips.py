#!/usr/bin/env python3
"""Build per-clip npz (with idle entry/exit) from every curated CSV."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = REPO / "data/clips"
BVH_DIR = Path("/Users/yumx/code/Beyondminic-Weilai-X1/downloads/lafan1")
# curated window offset inside the source BVH (only non-zero for windowed clips)
BVH_OFFSET = {"dance2_subject1_dance": 3918}

SKILL = {"walk": 1, "run": 2, "dance": 3}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csvs = sorted((REPO / "data/curated").glob("*.csv")) + sorted((REPO / "data/curated_umr").glob("*.csv"))
    for csv in csvs:
        stem = csv.stem
        gait = None
        for g in SKILL:
            if stem.endswith("_" + g):
                gait = g
                break
        if gait is None:
            print(f"[skip] {csv} (no gait suffix)")
            continue
        name = stem
        out = OUT / f"{name}.npz"
        if out.exists():
            print(f"[cache] {out.name}")
            continue
        fps = 120 if name.startswith("accad_") else 30
        cmd = [PY, "tools/build_clip_npz.py", "--csv", str(csv),
               "--out_npz", str(out), "--name", name,
               "--skill", str(SKILL[gait]), "--input_fps", str(fps)]
        if not (name.startswith("accad_") or name.startswith("eval_") or name.startswith("idle_")):
            base = name.rsplit("_", 1)[0]
            bvh = BVH_DIR / f"{base}.bvh"
            if bvh.exists():
                cmd += ["--bvh", str(bvh), "--bvh_offset", str(BVH_OFFSET.get(name, 0))]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
        if r.returncode != 0:
            print(r.stderr[-800:])


if __name__ == "__main__":
    main()
