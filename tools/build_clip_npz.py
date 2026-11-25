#!/usr/bin/env python3
"""Build a BeyondMimic-compatible F1 clip npz with an idle entry/exit.

Pipeline for one curated, contact-repaired CSV window (36 columns:
root xyz, root quat xyzw, 29 joints in F1_JOINT_NAMES order):

  [idle stand] -> [blend into the clip] -> [clip] -> [blend to stand] -> [idle]

The idle pose is the audited F1 stand (hip -0.2915, knee 0.450, ankle -0.1585,
base z 0.588).  The root XY stays at the clip's first frame so the robot does
not teleport; only joint angles and root height blend.

Output npz keys (all 50 Hz unless noted):
  fps, joint_pos (T,29), joint_vel, body_pos_w (T,30,3), body_quat_w (T,30,4 wxyz),
  body_lin_vel_w, body_ang_vel_w, body_names (30,), joint_names (29,),
  clip_name (str), skill_id (int), foot_contact (T,2), idle_mask (T,)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
F1_MJCF = REPO / "GMR/assets/f1/f1_mocap.xml"
F1_URDF = REPO / "F1_29DOF/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf"

F1_JOINT_NAMES = [
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
F1_BODY_ORDER = [
    "base_link",
    "left_hip_pitch", "right_hip_pitch", "lumbar_yaw_link",
    "left_hip_roll", "right_hip_roll", "lumbar_roll_link",
    "left_hip_yaw", "right_hip_yaw", "lumbar_pitch_link",
    "left_knee_pitch", "right_knee_pitch", "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch", "right_ankle_pitch", "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll", "right_ankle_roll", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_pitch_link", "right_elbow_pitch_link",
    "left_elbow_yaw_link", "right_elbow_yaw_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
]
FOOT_CORNERS = {
    "left": [(-0.0967, -0.0395, -0.0337), (0.1071, -0.0395, -0.0337),
             (-0.0967, 0.0365, -0.0337), (0.1071, 0.0365, -0.0337)],
    "right": [(-0.0967, -0.0365, -0.0337), (0.1071, -0.0365, -0.0337),
              (-0.0967, 0.0395, -0.0337), (0.1071, 0.0395, -0.0337)],
}
FOOT_SPHERE_R = 0.002


def stand_qpos() -> np.ndarray:
    d = json.load(open(REPO / "F1_29DOF/mjcf/robot/f1/f1_stand.json"))
    q = np.zeros(36)
    q[2] = 0.588
    # the CSV/motion array stores the root quaternion in xyzw order
    q[3:7] = [0, 0, 0, 1]
    for i, n in enumerate(F1_JOINT_NAMES):
        q[7 + i] = d["joint_pos_urdf_order"][n]
    return q


def slerp(q0_xyzw, q1_xyzw, t):
    r0 = R.from_quat(q0_xyzw)
    r1 = R.from_quat(q1_xyzw)
    return (r0 * (r0.inv() * r1) ** t).as_quat()


def build_with_idle(motion: np.ndarray, fps_in: int, idle_s: float, blend_s: float):
    n = len(motion)
    n_idle = int(round(idle_s * fps_in))
    n_blend = int(round(blend_s * fps_in))
    stand = stand_qpos()
    # root xy stays at the clip start; root z / quat blend from stand
    first = motion[0].copy()
    first[0:3] = [motion[0, 0], motion[0, 1], stand[2]]
    first[3:7] = [0, 0, 0, 1]
    last = motion[-1].copy()
    last[0:3] = [motion[-1, 0], motion[-1, 1], stand[2]]
    last[3:7] = [0, 0, 0, 1]

    pre = np.repeat(stand[None, :], n_idle, axis=0)
    pre[:, 0:2] = motion[0, 0:2]
    post = np.repeat(stand[None, :], n_idle, axis=0)
    post[:, 0:2] = motion[-1, 0:2]

    blend_pre = np.zeros((n_blend, 36))
    for k in range(n_blend):
        a = (k + 1) / (n_blend + 1)
        blend_pre[k, 0:2] = motion[0, 0:2]
        blend_pre[k, 2] = (1 - a) * stand[2] + a * first[2]
        blend_pre[k, 3:7] = slerp(stand[3:7], motion[0, 3:7], a)
        blend_pre[k, 7:] = (1 - a) * stand[7:] + a * motion[0, 7:]
    blend_post = np.zeros((n_blend, 36))
    for k in range(n_blend):
        a = (k + 1) / (n_blend + 1)
        blend_post[k, 0:2] = motion[-1, 0:2]
        blend_post[k, 2] = (1 - a) * motion[-1, 2] + a * stand[2]
        blend_post[k, 3:7] = slerp(motion[-1, 3:7], stand[3:7], a)
        blend_post[k, 7:] = (1 - a) * motion[-1, 7:] + a * stand[7:]

    out = np.concatenate([pre, blend_pre, motion, blend_post, post], axis=0)
    idle = np.concatenate([
        np.ones(n_idle), np.zeros(n_blend), np.zeros(n), np.zeros(n_blend), np.ones(n_idle)
    ])
    return out, idle


def resample(motion: np.ndarray, fps_in: int, fps_out: int = 50):
    n_in = len(motion)
    in_dt = 1.0 / fps_in
    out_dt = 1.0 / fps_out
    duration = (n_in - 1) * in_dt
    times = np.arange(0, duration, out_dt)
    phase = times / duration
    idx0 = np.floor(phase * (n_in - 1)).astype(int)
    idx1 = np.minimum(idx0 + 1, n_in - 1)
    blend = phase * (n_in - 1) - idx0

    pos = motion[idx0, 0:3] * (1 - blend[:, None]) + motion[idx1, 0:3] * blend[:, None]
    joints = motion[idx0, 7:] * (1 - blend[:, None]) + motion[idx1, 7:] * blend[:, None]
    qa = R.from_quat(motion[idx0, 3:7])
    qb = R.from_quat(motion[idx1, 3:7])
    quats = np.array([qa[i].as_quat() for i in range(len(blend))]) * (1 - blend[:, None]) + \
            np.array([qb[i].as_quat() for i in range(len(blend))]) * blend[:, None]
    quats = R.from_quat(quats).as_quat()[:, [3, 0, 1, 2]]  # wxyz
    return pos, quats, joints, out_dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out_npz", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--skill", type=int, required=True, choices=[0, 1, 2, 3],
                    help="0 idle, 1 walk, 2 run, 3 dance")
    ap.add_argument("--input_fps", type=int, default=30)
    ap.add_argument("--idle_s", type=float, default=2.0)
    ap.add_argument("--blend_s", type=float, default=0.8)
    ap.add_argument("--bvh", type=Path, default=None,
                    help="source LAFAN BVH; use its contact schedule for foot_contact")
    ap.add_argument("--bvh_offset", type=int, default=0)
    args = ap.parse_args()

    motion = np.loadtxt(args.csv, delimiter=",")
    assert motion.shape[1] == 36, motion.shape
    T_orig = len(motion)
    contact_in = None
    if args.bvh is not None:
        import sys as _sys
        _sys.path.insert(0, str(REPO / "GMR"))
        from general_motion_retargeting.utils.lafan1 import load_bvh_file
        frames, _ = load_bvh_file(str(args.bvh), format="lafan1")
        L = np.array([f["LeftFoot"][0] for f in frames])
        Rf = np.array([f["RightFoot"][0] for f in frames])
        vL = np.linalg.norm(np.diff(L, axis=0), axis=1)
        vR = np.linalg.norm(np.diff(Rf, axis=0), axis=1)
        cl = np.r_[vL < 0.02, False] & (L[:, 2] < 0.12)
        cr = np.r_[vR < 0.02, False] & (Rf[:, 2] < 0.12)
        c = np.stack([cl, cr], axis=1)[args.bvh_offset:args.bvh_offset + T_orig]
        assert len(c) == T_orig, (len(c), T_orig)
        n_idle = int(round(args.idle_s * args.input_fps))
        n_blend = int(round(args.blend_s * args.input_fps))
        contact_in = np.ones((len(motion) + 2 * n_idle + 2 * n_blend, 2), dtype=np.float32)
        contact_in[n_idle + n_blend:n_idle + n_blend + T_orig] = c
    motion, idle_mask = build_with_idle(motion, args.input_fps, args.idle_s, args.blend_s)
    pos, quats, joints, out_dt = resample(motion, args.input_fps)
    T = len(pos)
    # resample the idle mask to the output frame count (nearest input frame)
    src = np.minimum((np.arange(T) / max(T - 1, 1) * (len(idle_mask) - 1)).astype(int), len(idle_mask) - 1)
    idle_mask = idle_mask[src]
    contact_in = None if contact_in is None else contact_in[src]
    print(f"[clip] {args.name}: {len(motion)}f@{args.input_fps} -> {T}f@50 "
          f"(idle {int(idle_mask.sum())}f)")

    model = mujoco.MjModel.from_xml_path(str(F1_MJCF))
    data = mujoco.MjData(model)
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in F1_JOINT_NAMES}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in F1_JOINT_NAMES])
    bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in F1_BODY_ORDER}
    assert all(v >= 0 for v in bid.values()), [k for k, v in bid.items() if v < 0]
    foot_bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll")
                for s in ("left", "right")}

    body_pos = np.zeros((T, 30, 3))
    body_quat = np.zeros((T, 30, 4))
    foot_contact = np.zeros((T, 2), dtype=np.float32)

    def fk_pass():
        zmin = np.zeros(T)
        for t in range(T):
            mujoco.mj_resetData(model, data)
            data.qpos[0:3] = pos[t]
            data.qpos[3:7] = quats[t]
            data.qpos[qadr] = joints[t]
            mujoco.mj_forward(model, data)
            for i, name in enumerate(F1_BODY_ORDER):
                body_pos[t, i] = data.xpos[bid[name]]
                body_quat[t, i] = data.xquat[bid[name]]
            z = np.inf
            for j, s in enumerate(("left", "right")):
                b = foot_bid[s]
                Rm = data.xmat[b].reshape(3, 3)
                p = data.xpos[b]
                zs = min((p + Rm @ np.array(c))[2] - FOOT_SPHERE_R for c in FOOT_CORNERS[s])
                z = min(z, zs)
                foot_contact[t, j] = 1.0 if zs < 0.03 else 0.0
            zmin[t] = z
        return zmin

    zmin = fk_pass()
    # idle / blend frames can dip a foot below the floor (root z blends from
    # stand to a crouched first frame).  Lift the root smoothly so the clip is
    # ground-clean; this only affects the transition frames.
    lift_raw = np.maximum(0.0, -(zmin - 0.005))
    if lift_raw.max() > 0:
        x = np.arange(-10, 11)
        kern = np.exp(-0.5 * (x / 2.0) ** 2)
        kern /= kern.sum()
        lift = np.maximum(np.convolve(lift_raw, kern, mode="same"), lift_raw)
        pos[:, 2] += lift
        print(f"[clip] idle/blend ground lift: max {lift.max()*1000:.1f} mm on {int((lift_raw>0).sum())} frames")
        fk_pass()

    # idle frames: both feet planted by construction
    foot_contact[idle_mask > 0.5] = 1.0
    if contact_in is not None:
        # source contact must agree with the retargeted geometry on stance
        # frames; a mismatch means the BVH table is offset from the CSV window
        # and would poison contact_phase + the contact rewards.
        fk_contact = foot_contact.copy()
        stance = contact_in > 0.5
        if stance.any():
            agree = float((fk_contact[stance] == stance[stance]).mean())
            print(f"[contact] source-stance agreement {agree*100:.1f}% "
                  f"({int(stance.sum())} frames)")
            assert agree >= 0.90, (
                f"source contact disagrees with FK sole-z on stance frames "
                f"({agree*100:.1f}% < 90%): check the BVH window offset")
        foot_contact = contact_in.astype(np.float32)
        foot_contact[idle_mask > 0.5] = 1.0
    # the stand quaternion is xyzw identity; catch a wxyz/xyzw convention mixup
    assert np.allclose(body_quat[0, 0], [1.0, 0.0, 0.0, 0.0], atol=1e-5), body_quat[0, 0]

    joint_vel = np.gradient(joints, out_dt, axis=0)
    body_lin_vel = np.gradient(body_pos, out_dt, axis=0)
    q_prev, q_next = quats[:-2], quats[2:]
    r_prev = R.from_quat(q_prev[:, [1, 2, 3, 0]])
    r_next = R.from_quat(q_next[:, [1, 2, 3, 0]])
    omega = (r_next * r_prev.inv()).as_rotvec() / (2 * out_dt)
    omega = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
    body_ang_vel = np.broadcast_to(omega[:, None, :], (T, 30, 3)).copy()

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        fps=50,
        joint_pos=joints.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        body_names=np.array(F1_BODY_ORDER),
        joint_names=np.array(F1_JOINT_NAMES),
        clip_name=args.name,
        skill_id=args.skill,
        foot_contact=foot_contact,
        idle_mask=idle_mask.astype(np.float32),
    )
    print(f"[saved] {args.out_npz}  body z {body_pos[:,:,2].min():.3f}..{body_pos[:,:,2].max():.3f}")


if __name__ == "__main__":
    main()
