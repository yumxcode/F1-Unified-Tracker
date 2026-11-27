"""Extract a skill subset (e.g. walk-only, dance-only) from a motion library."""
import argparse
import numpy as np

TIME_FIELDS = ["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
               "joint_pos", "joint_vel", "foot_contact", "idle_mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--skill", type=int, required=True, help="skill id to keep")
    args = ap.parse_args()
    d = np.load(args.src)
    names = [str(x) for x in d["clip_names"]]
    starts = [int(x) for x in d["clip_start"]]
    ends = [int(x) for x in d["clip_end"]]
    skills = [int(x) for x in d["skill_ids"]]
    keep = [i for i, s in enumerate(skills) if s == args.skill]
    out = {k: [] for k in TIME_FIELDS}
    new_starts, new_ends, new_names, new_skills = [], [], [], []
    cursor = 0
    for i in keep:
        s, e = starts[i], ends[i]
        for k in TIME_FIELDS:
            out[k].append(d[k][s:e])
        n = e - s
        new_starts.append(cursor)
        new_ends.append(cursor + n)
        new_names.append(names[i])
        new_skills.append(skills[i])
        cursor += n
        print(f"keep {names[i]} [{s},{e}) {n}f")
    data = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    data.update({
        "fps": np.array([d["fps"]]),
        "clip_names": np.array(new_names),
        "clip_start": np.array(new_starts, dtype=np.int64),
        "clip_end": np.array(new_ends, dtype=np.int64),
        "skill_ids": np.array(new_skills, dtype=np.int64),
    })
    for k in d.files:
        if k not in data:
            data[k] = d[k]
    np.savez_compressed(args.dst, **data)
    print(f"wrote {args.dst}: {len(new_names)} clips, {cursor} frames")


if __name__ == "__main__":
    main()
