#!/usr/bin/env python3
"""Rebuild data/curated/curation_report.json from the per-clip gate reports."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CUR = REPO / "data/curated"


def main() -> None:
    report = {}
    for p in sorted(CUR.glob("*_*.json")):
        if p.name == "curation_report.json":
            continue
        d = json.load(open(p))
        name = p.stem
        # name is <clip>_<gait>; recover the gait from the suffix
        gait = name.rsplit("_", 1)[-1]
        report[name] = {
            "gait": gait,
            "pass": bool(d.get("overall_pass", False)),
            "frames": d.get("frames"),
            "csv": str(p.with_suffix(".csv")),
            "report": str(p),
        }
    out = CUR / "curation_report.json"
    out.write_text(json.dumps(report, indent=2))
    ok = sum(1 for v in report.values() if v["pass"])
    print(f"[curation] {ok}/{len(report)} clips pass -> {out}")


if __name__ == "__main__":
    main()
