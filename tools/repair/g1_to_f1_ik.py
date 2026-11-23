#!/usr/bin/env python3
"""Retarget the LAFAN1 -> Unitree G1 dataset to Weichai F1 via whole-body IK.

The public `lvhaidong/LAFAN1_Retargeting_Dataset` G1 CSVs are already
contact-consistent (Interaction-Mesh + IK with end-effector pose constraints
and anti-slippage), but they are expressed for G1 geometry and joint
conventions.  This script uses the G1 trajectory as a *kinematic reference*:
it forward-kinematics the G1 model, scales the body trajectories to F1 size,
and solves F1 whole-body IK per frame with frame tasks on both feet (flat
soles), the pelvis and both wrists.  A low-weight posture task keeps the
solution near the name-mapped G1 joint angles.

Output: a 36-column CSV in the BeyondMimic F1 layout
(root xyz, root quat xyzw, 29 joints in F1_JOINT_NAMES order).

Usage:
    python tools/repair/g1_to_f1_ik.py --g1_csv data/raw/lafan1_g1/walk1_subject1.csv \
        --out_csv motions/csv/walk1_subject1_f1.csv --stride 1
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
G1_XML = REPO / "whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
F1_XML = REPO / "GMR/assets/f1/f1_mocap.xml"

# CSV joint order (dataset README), 29 joints
G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

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

# name map G1 -> F1, plus sign flips for joints whose zero/range convention is
# mirrored between the two robots (calibrated from rest poses).
G1_TO_F1 = {n: n for n in G1_JOINTS}
G1_TO_F1.update({
    "left_knee_joint": "left_knee_pitch_joint",
    "right_knee_joint": "right_knee_pitch_joint",
    "waist_yaw_joint": "lumbar_yaw_joint",
    "waist_roll_joint": "lumbar_roll_joint",
    "waist_pitch_joint": "lumbar_pitch_joint",
    "left_elbow_joint": "left_elbow_pitch_joint",
    "right_elbow_joint": "right_elbow_pitch_joint",
    "left_wrist_yaw_joint": "left_elbow_yaw_joint",
    "right_wrist_yaw_joint": "right_elbow_yaw_joint",
})
SIGN = {n: 1.0 for n in G1_JOINTS}
# F1 shoulder-roll range is [-2, 0] (T-pose = -1.57); G1 uses +1.57.  Flip.
for n in ("left_shoulder_roll_joint", "right_shoulder_roll_joint"):
    SIGN[n] = -1.0

G1_BODY = {"left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link",
           "left_hand": "left_wrist_roll_link", "right_hand": "right_wrist_roll_link"}
F1_BODY = {"left_foot": "left_ankle_roll", "right_foot": "right_ankle_roll",
           "left_hand": "left_wrist_roll_link", "right_hand": "right_wrist_roll_link",
           "base": "base_link"}

F1_SOLE_OFFSET = 0.0337  # ankle_roll frame origin -> sole plane (m)
G1_SOLE_OFFSET = 0.0300  # approximate G1 ankle->sole


def rpy_to_so3(rpy) -> mink.SO3:
    return mink.SO3.from_matrix(R.from_euler("xyz", rpy).as_matrix())


def yaw_only_quat(quat_xyzw) -> mink.SO3:
    r = R.from_quat(quat_xyzw)
    yaw = r.as_euler("xyz")[2]
    return mink.SO3.from_matrix(R.from_euler("xyz", [0.0, 0.0, yaw]).as_matrix())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1_csv", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scale", type=float, default=None,
                    help="F1/G1 scale; default 0.588/0.797")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--solver", default="quadprog")
    ap.add_argument("--no_hands", action="store_true")
    args = ap.parse_args()

    raw = np.loadtxt(args.g1_csv, delimiter=",")
    assert raw.shape[1] == 36, raw.shape
    end = len(raw) if args.end is None else args.end
    raw = raw[args.start:end:args.stride]
    n = len(raw)
    s = args.scale if args.scale is not None else 0.588 / 0.797
    print(f"[g1->f1] {args.g1_csv.name}: {n} frames, scale {s:.4f}")

    # ---- G1 FK -------------------------------------------------------------
    g1_model = mujoco.MjModel.from_xml_path(str(G1_XML))
    g1_data = mujoco.MjData(g1_model)
    g1_qadr = np.array([g1_model.jnt_qposadr[
        mujoco.mj_name2id(g1_model, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in G1_JOINTS])
    g1_bid = {k: mujoco.mj_name2id(g1_model, mujoco.mjtObj.mjOBJ_BODY, v)
              for k, v in G1_BODY.items()}
    g1_pelvis = mujoco.mj_name2id(g1_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert g1_pelvis >= 0
    assert all(v >= 0 for v in g1_bid.values()), g1_bid

    # world trajectories of the G1 bodies (before scaling)
    g1_root = np.zeros((n, 3))
    g1_root_quat = np.zeros((n, 4))  # xyzw
    g1_pos = {k: np.zeros((n, 3)) for k in G1_BODY}
    g1_quat = {k: np.zeros((n, 4)) for k in G1_BODY}
    for i in range(n):
        q = np.zeros(g1_model.nq)
        q[0:3] = raw[i, 0:3]
        q[3:7] = [raw[i, 6], raw[i, 3], raw[i, 4], raw[i, 5]]  # xyzw -> wxyz
        q[g1_qadr] = raw[i, 7:]
        g1_data.qpos[:] = q
        mujoco.mj_forward(g1_model, g1_data)
        g1_root[i] = g1_data.xpos[g1_pelvis]
        g1_root_quat[i] = raw[i, 3:7]
        for k, b in g1_bid.items():
            g1_pos[k][i] = g1_data.xpos[b]
            g1_quat[k][i] = R.from_matrix(g1_data.xmat[b].reshape(3, 3)).as_quat()

    # scale around the initial root, anchor the F1 root at the nominal stand height
    root0 = g1_root[0].copy()
    f1_anchor = np.array([0.0, 0.0, 0.588])

    def scaled(p):
        return s * (p - root0) + f1_anchor

    # ---- F1 IK -------------------------------------------------------------
    f1_model = mujoco.MjModel.from_xml_path(str(F1_XML))
    f1_data = mujoco.MjData(f1_model)
    cfg = mink.Configuration(f1_model)
    # initial configuration: G1 name-mapped angles
    q0 = np.zeros(f1_model.nq)
    q0[3] = 1.0
    for j, nm in enumerate(G1_JOINTS):
        jid = mujoco.mj_name2id(f1_model, mujoco.mjtObj.mjOBJ_JOINT, G1_TO_F1[nm])
        assert jid >= 0, nm
        q0[f1_model.jnt_qposadr[jid]] = SIGN[nm] * raw[0, 7 + j]
    cfg.update(q0)
    posture = mink.PostureTask(f1_model, cost=1e-3)
    posture.set_target(q0)

    tasks = [
        mink.FrameTask("left_ankle_roll", "body", position_cost=200.0, orientation_cost=200.0),
        mink.FrameTask("right_ankle_roll", "body", position_cost=200.0, orientation_cost=200.0),
        mink.FrameTask("base_link", "body", position_cost=50.0, orientation_cost=50.0),
    ]
    if not args.no_hands:
        tasks += [
            mink.FrameTask("left_wrist_roll_link", "body", position_cost=20.0, orientation_cost=5.0),
            mink.FrameTask("right_wrist_roll_link", "body", position_cost=20.0, orientation_cost=5.0),
        ]
    limits = [mink.ConfigurationLimit(f1_model)]
    dt = f1_model.opt.timestep

    out = np.zeros((n, 36), dtype=np.float64)
    t0 = time.time()
    for i in range(n):
        # targets in F1 world
        p_base = scaled(g1_root[i])
        q_base = yaw_only_quat(g1_root_quat[i])
        # feet: keep the scaled ankle position, but lift/lower so the sole sits
        # at the G1-scaled sole height; force a flat sole (yaw only)
        targets = {}
        for side in ("left", "right"):
            p = scaled(g1_pos[f"{side}_foot"][i]).copy()
            # G1 ankle z -> sole z; F1 ankle z = sole + F1_SOLE_OFFSET
            sole = (g1_pos[f"{side}_foot"][i][2] - root0[2]) * s
            p[2] = f1_anchor[2] + sole + (F1_SOLE_OFFSET - G1_SOLE_OFFSET * s)
            targets[side] = (p, yaw_only_quat(g1_quat[f"{side}_foot"][i]))

        tasks[0].set_target(mink.SE3.from_rotation_and_translation(targets["left"][1], targets["left"][0]))
        tasks[1].set_target(mink.SE3.from_rotation_and_translation(targets["right"][1], targets["right"][0]))
        tasks[2].set_target(mink.SE3.from_rotation_and_translation(q_base, p_base))
        if not args.no_hands:
            for k, side in enumerate(("left", "right")):
                tasks[3 + k].set_target(mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(R.from_quat(g1_quat[f"{side}_hand"][i]).as_matrix()),
                    scaled(g1_pos[f"{side}_hand"][i])))

        for _ in range(args.iters):
            vel = mink.solve_ik(cfg, tasks, dt, solver=args.solver, damping=1e-3,
                                limits=limits, safety_break=False)
            cfg.integrate_inplace(vel, dt)
        q = cfg.data.qpos.copy()
        out[i, 0:3] = q[0:3]
        # wxyz -> xyzw
        out[i, 3:7] = q[[4, 5, 6, 3]]
        for j, nm in enumerate(F1_JOINTS):
            jid = mujoco.mj_name2id(f1_model, mujoco.mjtObj.mjOBJ_JOINT, nm)
            out[i, 7 + j] = q[f1_model.jnt_qposadr[jid]]
        if i % 300 == 0:
            print(f"  frame {i}/{n} ({time.time() - t0:.1f}s)")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out_csv, out, delimiter=",", fmt="%.8f")
    print(f"[saved] {args.out_csv} ({n} frames, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
