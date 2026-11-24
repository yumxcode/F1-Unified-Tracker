#!/usr/bin/env python3
"""Contact-consistent kinematic repair for retargeted F1 motions.

Retargeters (GMR/UMR) map the human foot *pitch* onto the robot foot.  A human
foot is naturally toe-down ~20-25 deg at contact; the F1 foot is a rigid 20 cm
plate, so the toe digs into the floor (observed -6.6 cm) and the swing foot
never clears the ground (sole max +3.7 cm).  This pass re-solves whole-body IK
per frame with:

  * both ankle frames forced **flat** (yaw-only orientation),
  * stance ankle **locked** to its pose at contact onset (anti-skating),
  * swing ankle lifted so the sole clears the floor,
  * a low-weight base task preserving the retargeted root trajectory,
  * a low-weight posture task staying near the retargeted joint angles,
  * a per-joint velocity limit from the URDF.

The contact schedule is taken from the *source mocap* when --bvh is given
(LAFAN1 velocity threshold 0.02 m/frame), so a low retargeted foot is not
mistaken for stance.

Usage:
    python tools/repair/contact_repair.py --csv in.csv --out_csv out.csv \
        --bvh downloads/lafan1/walk1_subject1.bvh --start 0 --end 4918
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mink
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
F1_XML = REPO / "GMR/assets/f1/f1_mocap.xml"

F1_JOINTS = [
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
F1_BODY = {"left": "left_ankle_roll", "right": "right_ankle_roll", "base": "base_link"}
F1_SOLE_OFFSET = 0.0337  # ankle frame origin -> sole plane
ARM_TOKENS = ("shoulder", "elbow", "wrist")
ARM_JOINTS = [n for n in F1_JOINTS if any(t in n for t in ARM_TOKENS)]


def yaw_only(quat_xyzw) -> mink.SO3:
    yaw = R.from_quat(quat_xyzw).as_euler("xyz")[2]
    return mink.SO3.from_matrix(R.from_euler("xyz", [0.0, 0.0, yaw]).as_matrix())


def load_bvh_contacts(bvh: Path, n: int, start: int, end: int, stride: int,
                      velfactor: float = 0.002, offset: int = 0) -> np.ndarray:
    sys.path.insert(0, str(REPO / "GMR"))
    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    frames, _ = load_bvh_file(str(bvh), format="lafan1")
    L = np.array([f["LeftFoot"][0] for f in frames])
    Rf = np.array([f["RightFoot"][0] for f in frames])
    vL = np.linalg.norm(np.diff(L, axis=0), axis=1)
    vR = np.linalg.norm(np.diff(Rf, axis=0), axis=1)
    cl = np.r_[vL < velfactor, False]
    cr = np.r_[vR < velfactor, False]
    c = np.stack([cl, cr], axis=1)[offset + start:offset + end:stride]
    assert len(c) == n, (len(c), n)
    return c


def smooth_runs(c: np.ndarray, min_len: int = 3) -> np.ndarray:
    out = c.copy()
    for k in range(c.shape[1]):
        x = c[:, k]
        i = 0
        while i < len(x):
            if x[i]:
                j = i
                while j < len(x) and x[j]:
                    j += 1
                if j - i < min_len:
                    out[i:j, k] = False
                i = j
            else:
                i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--bvh", type=Path, default=None,
                    help="source LAFAN1 BVH; enables source contact schedule + foot lock")
    ap.add_argument("--bvh_offset", type=int, default=0,
                    help="frame offset of the CSV inside the BVH (for pre-windowed CSVs)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--solver", default="quadprog")
    ap.add_argument("--stance_h", type=float, default=0.055,
                    help="ankle z below this = stance (fallback only)")
    ap.add_argument("--swing_clear", type=float, default=0.05,
                    help="minimum sole clearance during swing (m)")
    ap.add_argument("--flat_cost", type=float, default=300.0)
    ap.add_argument("--foot_cost", type=float, default=200.0,
                    help="foot position cost in the IK")
    ap.add_argument("--base_cost", type=float, default=5.0)
    ap.add_argument("--posture_cost", type=float, default=1e-3)
    ap.add_argument("--vel_scale", type=float, default=0.8,
                    help="fraction of the URDF velocity limit enforced by the IK")
    ap.add_argument("--ground_rounds", type=int, default=2,
                    help="max ground-correction rounds after the main IK")
    ap.add_argument("--stance_widen_repair", type=float, default=0.0,
                    help="shift foot targets laterally by this much per side (m)")
    ap.add_argument("--no_post_lift", action="store_true",
                    help="disable the final root-z ground-clearance pass")
    ap.add_argument("--no_lock", action="store_true",
                    help="do not lock the stance foot at contact onset")
    ap.add_argument("--velfactor", type=float, default=0.02,
                    help="source contact velocity threshold (m/frame); must match the clip labels")
    args = ap.parse_args()

    raw = np.loadtxt(args.csv, delimiter=",")
    assert raw.shape[1] == 36, raw.shape
    end = len(raw) if args.end is None else args.end
    raw = raw[args.start:end:args.stride]
    n = len(raw)
    print(f"[repair] {args.csv.name}: {n} frames")

    if args.bvh is not None:
        contact = smooth_runs(load_bvh_contacts(args.bvh, n, args.start, end, args.stride,
                                                velfactor=args.velfactor,
                                                offset=args.bvh_offset))
        print(f"[repair] source contact: DS {np.mean(contact[:,0] & contact[:,1]):.2f} "
              f"L {contact[:,0].mean():.2f} R {contact[:,1].mean():.2f}")
    else:
        contact = None

    model = mujoco.MjModel.from_xml_path(str(F1_XML))
    data = mujoco.MjData(model)
    cfg = mink.Configuration(model)
    bid = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, v) for k, v in F1_BODY.items()}
    assert all(v >= 0 for v in bid.values()), bid
    qadr = np.array([model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in F1_JOINTS])

    tasks = [
        mink.FrameTask(F1_BODY["left"], "body", position_cost=args.foot_cost, orientation_cost=args.flat_cost),
        mink.FrameTask(F1_BODY["right"], "body", position_cost=args.foot_cost, orientation_cost=args.flat_cost),
        mink.FrameTask(F1_BODY["base"], "body", position_cost=args.base_cost, orientation_cost=args.base_cost),
    ]
    posture = mink.PostureTask(model, cost=args.posture_cost)
    limits = [mink.ConfigurationLimit(model)]
    sys.path.insert(0, str(REPO))
    from tools_check_motion_f1 import urdf_limits, FOOT_CORNERS, FOOT_SPHERE_R
    if args.vel_scale > 0:
        ul = urdf_limits()
        vlim = {nm: args.vel_scale * ul[nm][2] for nm in F1_JOINTS}
        limits.append(mink.VelocityLimit(model, vlim))
    dt = model.opt.timestep

    out = np.zeros((n, 36), dtype=np.float64)
    lock = {s: None for s in ("left", "right")}
    prev_contact = {s: False for s in ("left", "right")}
    t0 = time.time()
    for i in range(n):
        q = np.zeros(model.nq)
        q[0:3] = raw[i, 0:3]
        qq = raw[i, 3:7] / np.linalg.norm(raw[i, 3:7])
        q[3:7] = [qq[3], qq[0], qq[1], qq[2]]
        q[qadr] = raw[i, 7:]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)

        targets = {}
        root_yaw = R.from_quat(raw[i, 3:7]).as_euler("xyz")[2]
        lateral = np.array([-np.sin(root_yaw), np.cos(root_yaw), 0.0])
        for side in ("left", "right"):
            b = bid[side]
            p = data.xpos[b].copy()
            if args.stance_widen_repair:
                p = p + (1.0 if side == "left" else -1.0) * args.stance_widen_repair * lateral
            quat = R.from_matrix(data.xmat[b].reshape(3, 3)).as_quat()
            if contact is not None:
                stance = bool(contact[i, 0 if side == "left" else 1])
            else:
                stance = p[2] < args.stance_h
            if stance:
                if lock[side] is None:
                    lock[side] = (p.copy(), yaw_only(quat))
                if args.no_lock:
                    lock[side] = (p.copy(), yaw_only(quat))
                lp, lq = lock[side]
                p = np.array([lp[0], lp[1], F1_SOLE_OFFSET + 0.001])
                targets[side] = (p, lq, True)
            else:
                lock[side] = None
                p[2] = max(p[2], F1_SOLE_OFFSET + args.swing_clear)
                targets[side] = (p, yaw_only(quat), False)

        cfg.update(q)
        posture.set_target(q)
        tasks[0].set_target(mink.SE3.from_rotation_and_translation(targets["left"][1], targets["left"][0]))
        tasks[1].set_target(mink.SE3.from_rotation_and_translation(targets["right"][1], targets["right"][0]))
        base_p = q[0:3].copy()
        base_q = yaw_only(raw[i, 3:7])
        tasks[2].set_target(mink.SE3.from_rotation_and_translation(base_q, base_p))

        for _ in range(args.iters):
            vel = mink.solve_ik(cfg, [tasks[0], tasks[1], tasks[2], posture], dt,
                                solver=args.solver, damping=1e-3, limits=limits,
                                safety_break=False)
            cfg.integrate_inplace(vel, dt)

        # ground correction: keep the stance foot planted and the swing foot
        # clear **locally** -- never lift the whole root (that floats the
        # stance foot and poisons the contact labels).
        for _round in range(args.ground_rounds):
            mujoco.mj_forward(model, cfg.data)
            changed = False
            for side in ("left", "right"):
                b = bid[side]
                Rm = cfg.data.xmat[b].reshape(3, 3)
                p = cfg.data.xpos[b]
                z = min((p + Rm @ np.array(c))[2] - FOOT_SPHERE_R for c in FOOT_CORNERS[side])
                if targets[side][2]:
                    # stance: the sole must sit at ~5 mm.  If the leg cannot
                    # reach it, lower the base target for this frame instead
                    # of lifting the whole body.
                    if z > 0.008:
                        base_p[2] = max(0.35, base_p[2] - (z - 0.005))
                        tasks[2].set_target(
                            mink.SE3.from_rotation_and_translation(base_q, base_p))
                        changed = True
                    elif z < -0.005:
                        targets[side] = (targets[side][0] + np.array([0, 0, -z + 0.005]),
                                         targets[side][1], True)
                        tasks[0 if side == "left" else 1].set_target(
                            mink.SE3.from_rotation_and_translation(targets[side][1], targets[side][0]))
                        changed = True
                else:
                    # swing: lift only this foot's target
                    if z < 0.005:
                        targets[side] = (targets[side][0] + np.array([0, 0, 0.005 - z]),
                                         targets[side][1], False)
                        tasks[0 if side == "left" else 1].set_target(
                            mink.SE3.from_rotation_and_translation(targets[side][1], targets[side][0]))
                        changed = True
            if not changed:
                break
            for _ in range(10):
                vel = mink.solve_ik(cfg, [tasks[0], tasks[1], tasks[2], posture], dt,
                                    solver=args.solver, damping=1e-3, limits=limits,
                                    safety_break=False)
                cfg.integrate_inplace(vel, dt)

        qq = cfg.data.qpos.copy()
        # keep the retargeted arm posture: the repair only owns legs + root
        for j, nm in enumerate(F1_JOINTS):
            if nm in ARM_JOINTS:
                qq[qadr[j]] = raw[i, 7 + j]
        out[i, 0:3] = qq[0:3]
        out[i, 3:7] = qq[[4, 5, 6, 3]]
        for j, nm in enumerate(F1_JOINTS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
            out[i, 7 + j] = qq[model.jnt_qposadr[jid]]
        if i % 500 == 0:
            print(f"  frame {i}/{n} ({time.time() - t0:.1f}s)")

    # ---- joint velocity clamp (post-IK) ----------------------------------
    # A strong foot lock can ask for a knee/ankle step larger than the motor
    # limit at a contact transition.  Clamp the per-frame joint delta to the
    # URDF limit (scaled) and let the foot lag for 1-2 frames instead of
    # emitting an infeasible velocity.
    if args.vel_scale > 0:
        from tools_check_motion_f1 import urdf_limits as _ul
        ul = _ul()
        vmax = np.array([args.vel_scale * ul[n][2] for n in F1_JOINTS])
        dt_in = 1.0 / args.fps
        for i in range(1, n):
            dq = out[i, 7:] - out[i - 1, 7:]
            v = np.abs(dq) / dt_in
            over = v > vmax
            if np.any(over):
                scale = np.ones_like(dq)
                scale[over] = vmax[over] * dt_in / np.abs(dq[over])
                out[i, 7:] = out[i - 1, 7:] + dq * scale

    # ---- final ground-clearance pass -------------------------------------
    # A single unreachable landing frame can still leave a sole corner 1-4 cm
    # below the floor.  Lift the root z smoothly by exactly the deficit; this
    # keeps the sole >= -5 mm and is within the 1 cm floating tolerance.
    if not args.no_post_lift:
        zmin = np.zeros(n)
        for i in range(n):
            q = np.zeros(model.nq)
            q[0:3] = out[i, 0:3]
            qq = out[i, 3:7] / np.linalg.norm(out[i, 3:7])
            q[3:7] = [qq[3], qq[0], qq[1], qq[2]]
            q[qadr] = out[i, 7:]
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            z = np.inf
            for side in ("left", "right"):
                b = bid[side]
                Rm = data.xmat[b].reshape(3, 3)
                pp = data.xpos[b]
                for c in FOOT_CORNERS[side]:
                    z = min(z, (pp + Rm @ np.array(c))[2] - FOOT_SPHERE_R)
            zmin[i] = z
        lift_raw = np.maximum(0.0, -(zmin - 0.005))
        if lift_raw.max() > 0:
            x = np.arange(-10, 11)
            kern = np.exp(-0.5 * (x / 2.0) ** 2)
            kern /= kern.sum()
            lift = np.convolve(lift_raw, kern, mode="same")
            lift = np.maximum(lift, lift_raw)
            out[:, 2] += lift
            print(f"[repair] post-lift: max {lift.max()*1000:.1f} mm on {int((lift_raw>0).sum())} frames")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out_csv, out, delimiter=",", fmt="%.8f")
    print(f"[saved] {args.out_csv} ({n} frames, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
