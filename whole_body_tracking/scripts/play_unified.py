#!/usr/bin/env python3
"""Sim2sim playback of the unified F1 tracker from a standing start to a stop.

Unlike play_mujoco.py (which loops a single clip and resets to frame 0), this
script evaluates one clip of the multi-clip library through the full contract
protocol:

    3 s idle stand -> 0.5 s settle -> motion -> 2 s stop hold

It never resets the reference mid-episode and never loops.  It reports entry,
in-motion and exit success separately, plus tracking / contact / slip / flight
metrics.

Usage:
    python whole_body_tracking/scripts/play_unified.py \
        --library motions/f1_unified_library.npz \
        --checkpoint trained_models/unified_model_XXXX.pt \
        --clip walk1_subject1 --headless --metrics out.npz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import mujoco

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
import play_mujoco as pm  # noqa: E402

pm._ROBOT = "f1"

F1_XML = REPO / "whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/f1/mjcf/f1_flat.xml"
FOOT_CORNERS = [(-0.0967, -0.0395, -0.0337), (0.1071, -0.0395, -0.0337),
                (-0.0967, 0.0365, -0.0337), (0.1071, 0.0365, -0.0337)]

# Ground-truth F1 PhysX joint order (TASK_20261105_507 / play_mujoco.py).
# The policy obs and action are in this order; the npz stores F1_JOINT_NAMES.
F1_PHYSX_JOINT_ORDER = [
    "left_hip_pitch_joint", "lumbar_yaw_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "lumbar_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "lumbar_pitch_joint", "right_hip_yaw_joint",
    "left_knee_pitch_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "right_knee_pitch_joint", "left_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "right_ankle_roll_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]


def build_obs(t, lib, data, joint_pos_npz, joint_vel_npz, last_action, default_pos,
              body_ids, anchor_idx, to_px, legacy_obs=False):
    ref_jpos = lib["joint_pos"][t]
    ref_jvel = lib["joint_vel"][t]
    ref_body_pos = lib["body_pos_w"][t]
    ref_body_quat = lib["body_quat_w"][t]
    robot_anchor_pos = data.xpos[body_ids[anchor_idx]].copy()
    robot_anchor_quat = data.xquat[body_ids[anchor_idx]].copy()
    anchor_pos_b, anchor_quat_b = pm.subtract_frame_transform(
        robot_anchor_pos, robot_anchor_quat,
        ref_body_pos[anchor_idx], ref_body_quat[anchor_idx])
    rotmat_b = pm.quat_to_rotmat(anchor_quat_b)
    base_lin_vel = pm.quat_rotate_inv(data.qpos[3:7], data.qvel[0:3].copy())
    base_ang_vel = data.qvel[3:6].copy()
    skill = np.zeros(4, dtype=np.float32)
    skill[int(lib["skill_ids"][lib["_clip_idx"]])] = 1.0
    contact = lib["foot_contact"][t]
    flight = (1.0 - contact[0]) * (1.0 - contact[1])
    parts = [
        to_px(ref_jpos), to_px(ref_jvel),
        anchor_pos_b, rotmat_b[:, :2].flatten(),
        base_lin_vel, base_ang_vel,
        to_px(joint_pos_npz - default_pos), to_px(joint_vel_npz), last_action,
    ]
    if not legacy_obs:
        parts += [skill, np.array([contact[0], contact[1], flight], dtype=np.float32)]
    obs_np = np.concatenate(parts).astype(np.float32)
    return obs_np, ref_body_pos, ref_body_quat, robot_anchor_pos, robot_anchor_quat, anchor_pos_b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--clip", type=str, required=True)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--metrics", type=Path, default=None)
    ap.add_argument("--json_out", type=Path, default=None)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = full clip")
    ap.add_argument("--legacy_obs", action="store_true", help="160-D obs (no skill/contact)")
    ap.add_argument("--entry_s", type=float, default=3.5)
    ap.add_argument("--exit_s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0, help="perturbs the initial state (3 seeds x 2 replays)")
    ap.add_argument("--action_noise", type=float, default=0.0, help="Gaussian action noise std (rep2)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    lib = dict(np.load(args.library, allow_pickle=True))
    clip_names = [str(x) for x in lib["clip_names"]]
    if args.clip not in clip_names:
        raise SystemExit(f"clip {args.clip!r} not in library; available: {clip_names}")
    ci = clip_names.index(args.clip)
    lib["_clip_idx"] = ci
    start, end = int(lib["clip_start"][ci]), int(lib["clip_end"][ci])
    print(f"[unified] clip {args.clip} [{start},{end}) skill {int(lib['skill_ids'][ci])} "
          f"({(end-start)/50:.1f}s)")

    actor, normalizer, obs_dim, action_dim = pm.load_policy(str(args.checkpoint))
    print(f"[policy] obs_dim={obs_dim} action_dim={action_dim}")
    expected_obs = 160 if args.legacy_obs else 167
    assert obs_dim == expected_obs, f"policy obs_dim {obs_dim} != {expected_obs}"
    assert action_dim == 29, f"policy action_dim {action_dim} != 29"

    model = mujoco.MjModel.from_xml_path(str(F1_XML))
    # training uses sim dt 0.005 with decimation 4 (policy 50 Hz); the MJCF
    # defaults to 0.001, which would make the PD loop 5x slower than training
    model.opt.timestep = pm.SIM_DT
    data = mujoco.MjData(model)
    joint_names_px = [str(x) for x in lib["joint_names"]]
    body_names = [str(x) for x in lib["body_names"]]
    # npz column order (F1_JOINT_NAMES) -> policy PhysX order
    assert set(joint_names_px) == set(F1_PHYSX_JOINT_ORDER), "joint name set mismatch"
    npz_to_physx = np.array([joint_names_px.index(n) for n in F1_PHYSX_JOINT_ORDER])
    physx_to_npz = np.argsort(npz_to_physx)

    def to_px(v):
        return v[npz_to_physx]

    def from_px(v):
        return v[physx_to_npz]

    # boundary self-check: a one-hot action in PhysX order must move exactly
    # the same-named joint in the npz/PD order
    for k, name in enumerate(F1_PHYSX_JOINT_ORDER):
        a = np.zeros(29, dtype=np.float32)
        a[k] = 1.0
        a_npz = from_px(a)
        moved = np.where(a_npz != 0)[0]
        assert len(moved) == 1 and joint_names_px[moved[0]] == name, (name, moved)
    print("[perm] npz->PhysX self-check passed (29 joints, same-name mapping)")
    # map npz joint/body order to the MuJoCo model
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names_px}
    assert all(v >= 0 for v in jid.values()), [k for k, v in jid.items() if v < 0]
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in joint_names_px])
    dadr = np.array([model.jnt_dofadr[jid[n]] for n in joint_names_px])
    # actuator order is the MJCF joint order, NOT the npz joint order
    act_for_joint = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{n}")
                              for n in joint_names_px])
    assert all(a >= 0 for a in act_for_joint), [n for n, a in zip(joint_names_px, act_for_joint) if a < 0]
    bid = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names])
    assert all(v >= 0 for v in bid), [n for n, b in zip(body_names, bid) if b < 0]
    anchor_idx = body_names.index("lumbar_pitch_link")

    default_pos, kp, kd, action_scale = pm.build_joint_arrays(joint_names_px)

    # Align MuJoCo joint dynamics with the IsaacLab training model: the F1 MJCF
    # carries vendor damping/frictionloss and no armature, while training used
    # zero damping/friction and IsaacLab-configured armature.
    for n in joint_names_px:
        for pat, _effort, arm, _kp, _kd in pm._F1_JOINT_CLASS:
            if re.fullmatch(pat, n):
                d = model.jnt_dofadr[jid[n]]
                model.dof_armature[d] = arm
                model.dof_damping[d] = 0.0
                model.dof_frictionloss[d] = 0.0
                break
    print("[dynamics] F1 armature/damping aligned with IsaacLab training model")

    # initial state: the reference frame at clip_start (idle stand for the
    # unified clips; the reference pose for legacy single-clip libraries)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = lib["body_pos_w"][start, 0]
    data.qpos[3:7] = lib["body_quat_w"][start, 0]
    data.qpos[qadr] = lib["joint_pos"][start]
    data.qvel[0:3] = lib["body_lin_vel_w"][start, 0]
    data.qvel[3:6] = lib["body_ang_vel_w"][start, 0]
    data.qvel[dadr] = lib["joint_vel"][start]
    # seed perturbation of the initial joint state (the 3-seed protocol):
    # a small deterministic joint-space perturbation, base untouched
    if args.seed:
        data.qpos[qadr] = data.qpos[qadr] + 0.01 * rng.standard_normal(len(qadr))
        data.qvel[dadr] = data.qvel[dadr] + 0.05 * rng.standard_normal(len(dadr))
    mujoco.mj_forward(model, data)

    last_action = np.zeros(len(joint_names_px), dtype=np.float32)
    metrics = {k: [] for k in [
        "body_pos_err_rel", "body_pos_err_abs", "body_lin_vel_err",
        "joint_pos_err", "anchor_pos_err", "base_z",
        "action_rate", "contact_mismatch", "slip", "flight_ref", "flight_robot",
        "robot_xy", "ref_xy", "skill", "tilt",
        "sole_z_left", "sole_z_right", "foot_xy_left", "foot_xy_right", "com_xy",
        "slip_stance",
    ]}
    fell = False
    fall_step = None
    fall_reason = None
    n_steps = (end - start) if args.max_steps <= 0 else min(args.max_steps, end - start)
    t0 = time.time()
    recorder = None
    if args.record:
        import imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [0, 0, 0.6]
        cam.distance = 3.5
        cam.azimuth = 135
        cam.elevation = -15
        writer = imageio.get_writer(str(args.record), fps=50)
        recorder = (renderer, cam, writer)

    for step in range(n_steps):
        t = start + step
        joint_pos = np.array([data.qpos[a] for a in qadr])
        joint_vel = np.array([data.qvel[a] for a in dadr])
        obs_np, ref_body_pos, ref_body_quat, robot_anchor_pos, robot_anchor_quat, anchor_pos_b = \
            build_obs(t, lib, data, joint_pos, joint_vel, last_action, default_pos,
                      bid, anchor_idx, to_px, args.legacy_obs)
        with torch.no_grad():
            action = actor(pm.normalize_obs(torch.from_numpy(obs_np).unsqueeze(0), normalizer)).squeeze(0).numpy().astype(np.float32)
        if args.action_noise > 0:
            action = action + args.action_noise * rng.standard_normal(action.shape).astype(np.float32)
        metrics["action_rate"].append(float(np.mean(np.abs(action - last_action))))
        last_action = action.copy()
        action_npz = from_px(action)
        target_pos = default_pos + action_npz * action_scale
        for _ in range(pm.DECIMATION):
            jp = np.array([data.qpos[a] for a in qadr])
            jv = np.array([data.qvel[a] for a in dadr])
            tau = kp * (target_pos - jp) - kd * jv
            for i, n in enumerate(joint_names_px):
                eff, _, _ = pm._match_joint_params_auto(n)
                tau[i] = np.clip(tau[i], -eff, eff)
            ctrl = np.zeros(model.nu)
            ctrl[act_for_joint] = tau
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

        # ---- metrics -----------------------------------------------------
        robot_body_pos = data.xpos[bid]
        delta_ori = pm.yaw_quat(pm.quat_mul(robot_anchor_quat, pm.quat_inv(ref_body_quat[anchor_idx])))
        delta_pos = robot_anchor_pos.copy()
        delta_pos[2] = ref_body_pos[anchor_idx][2]
        ref_rel = delta_pos[None, :] + (ref_body_pos - ref_body_pos[anchor_idx][None, :]) @ pm.quat_to_rotmat(delta_ori).T
        metrics["body_pos_err_rel"].append(float(np.mean(np.linalg.norm(robot_body_pos - ref_rel, axis=1))))
        metrics["body_pos_err_abs"].append(float(np.mean(np.linalg.norm(robot_body_pos - ref_body_pos, axis=1))))
        metrics["body_lin_vel_err"].append(float(np.mean(np.linalg.norm(
            data.cvel[bid][:, 3:6] - lib["body_lin_vel_w"][t], axis=1))))
        metrics["joint_pos_err"].append(float(np.mean(np.abs(joint_pos - lib["joint_pos"][t]))))
        metrics["anchor_pos_err"].append(float(np.linalg.norm(anchor_pos_b)))
        metrics["base_z"].append(float(data.qpos[2]))
        qw, qx, qy, qz = data.qpos[3:7]
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        tilt_deg = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        metrics["tilt"].append(tilt_deg)
        metrics["robot_xy"].append(data.qpos[0:2].copy())
        metrics["ref_xy"].append(ref_body_pos[0, :2].copy())
        metrics["skill"].append(int(lib["skill_ids"][ci]))

        # foot contact from sole corners
        robot_contact = []
        for side in ("left", "right"):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll")
            Rm = data.xmat[b].reshape(3, 3)
            p = data.xpos[b]
            z = min((p + Rm @ np.array(c))[2] for c in FOOT_CORNERS)
            robot_contact.append(z < 0.03)
            metrics[f"sole_z_{side}"].append(float(z))
            metrics[f"foot_xy_{side}"].append(p[:2].copy())
        metrics["com_xy"].append(data.subtree_com[0][:2].copy())
        ref_contact = lib["foot_contact"][t] > 0.5
        mismatch = int(robot_contact[0] != ref_contact[0]) + int(robot_contact[1] != ref_contact[1])
        metrics["contact_mismatch"].append(mismatch)
        metrics["flight_ref"].append(int(not ref_contact[0] and not ref_contact[1]))
        metrics["flight_robot"].append(int(not robot_contact[0] and not robot_contact[1]))

        # slip: foot horizontal speed while reference stance
        if step > 0:
            dt = pm.POLICY_DT
            slip = 0.0
            for side in ("left", "right"):
                b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll")
                v = np.linalg.norm((data.xpos[b][:2] - metrics.get(f"_prev_{side}", data.xpos[b][:2])) / dt)
                metrics[f"_prev_{side}"] = data.xpos[b][:2].copy()
                if ref_contact[0 if side == "left" else 1]:
                    slip = max(slip, v)
            metrics["slip"].append(slip)
            metrics["slip_stance"].append(slip if (ref_contact[0] or ref_contact[1]) else np.nan)

        # frozen STANDARDS B2: root z < 0.55 x standing OR root tilt > 45 deg
        if fall_step is None and (data.qpos[2] < 0.55 * pm.F1_INIT_HEIGHT or tilt_deg > 45.0):
            fell = True
            fall_step = step
            fall_reason = "tilt" if tilt_deg > 45.0 else "height"
        if recorder is not None:
            recorder[0].update_scene(data, camera=recorder[1])
            recorder[2].append_data(recorder[0].render())

    if recorder is not None:
        recorder[2].close()
        print(f"[record] {args.record}")

    # ---- summary ---------------------------------------------------------
    n_entry = int(args.entry_s / pm.POLICY_DT)
    n_exit = int(args.exit_s / pm.POLICY_DT)
    # ---- entry sub-reasons (STANDARDS v1.3 operational definition) ----
    # entry window must start at the blend end (motion start), not at a fixed
    # frame: a time-scaled library stretches idle+blend, so a fixed 140-175
    # window would fall inside the blend and judge an empty (both-feet-down) window.
    # blend end = idle_end * 1.4: the idle prefix is 2 s and the blend 0.8 s
    # (ratio 1.4), so this is scale-invariant for time-scaled libraries
    # (original: 100 -> 140; slow07: 143 -> 200)
    idle = np.array(lib["idle_mask"][start:start + n_steps]) > 0.5
    nz = np.where(~idle)[0]
    idle_end = int(nz[0]) if len(nz) else 0
    blend_end = int(round(idle_end * 1.4))
    post_lo = blend_end
    post_hi = min(blend_end + int(0.7 / pm.POLICY_DT), len(metrics["contact_mismatch"]))
    entry_fall = bool(fall_step is not None and fall_step < post_hi)
    mm = np.array(metrics["contact_mismatch"], dtype=float)
    entry_mismatch_post_pct = float(np.mean(mm[post_lo:post_hi]) / 2 * 100) if post_hi > post_lo else 100.0
    # first reference single-support frame and first robot foot lift
    ref_c = lib["foot_contact"][start:start + n_steps] > 0.5
    ss_frames = np.where((ref_c[:, 0] != ref_c[:, 1]))[0]
    ss_frames = ss_frames[(ss_frames >= post_lo) & (ss_frames < post_hi)]
    entry_first_ss = int(ss_frames[0]) if len(ss_frames) else -1
    sz = np.maximum(np.array(metrics["sole_z_left"]), np.array(metrics["sole_z_right"]))
    lift = np.where(sz[post_lo:min(post_hi, len(sz))] > 0.03)[0]
    entry_first_lift = int(lift[0] + post_lo) if len(lift) else -1
    # CoM outside the double-support polygon while both feet are in contact
    com_out_frames = 0
    fx = {side: np.array(metrics[f"foot_xy_{side}"]) for side in ("left", "right")}
    com = np.array(metrics["com_xy"])
    for st in range(post_lo, min(post_hi, n_steps)):
        if ref_c[st, 0] and ref_c[st, 1] and mm[st] == 0:
            lp, rp = fx["left"][st], fx["right"][st]
            lo = np.minimum(lp, rp) - 0.05
            hi = np.maximum(lp, rp) + 0.05
            if not (lo[0] <= com[st, 0] <= hi[0] and lo[1] <= com[st, 1] <= hi[1]):
                com_out_frames += 1
    entry_com_out_ds = int(com_out_frames)
    entry_ok = (not entry_fall) and entry_mismatch_post_pct <= 1.0 and entry_com_out_ds == 0
    entry_ok_old = ((fall_step is None or fall_step >= n_entry)
                    and float(np.mean(metrics["contact_mismatch"][:n_entry]) / 2) <= 0.01)
    # the fall window for entry_ok must cover the scaled entry (blend + 0.7 s)
    entry_ok = entry_ok and (fall_step is None or fall_step >= post_hi)
    robot_xy = np.array(metrics["robot_xy"])
    exit_drift = float(np.linalg.norm(robot_xy[-1] - robot_xy[max(0, len(robot_xy) - n_exit)]))
    # exit success requires no fall anywhere in the episode (a fall before the
    # exit window must not be counted as a successful stop)
    exit_ok = (fall_step is None) and exit_drift <= 0.05
    body_err = np.array(metrics["body_pos_err_rel"])
    anchor_err = float(np.mean(metrics["anchor_pos_err"]))
    success = (not fell) and anchor_err < 0.5
    out = {
        "clip": args.clip,
        "checkpoint": str(args.checkpoint),
        "library": str(args.library),
        "skill": int(lib["skill_ids"][ci]),
        "frames": n_steps,
        "seconds": round(n_steps / 50.0, 2),
        "falls": bool(fell),
        "fall_reason": fall_reason,
        "fall_step": fall_step,
        "fall_time_s": round(fall_step / 50.0, 2) if fall_step is not None else None,
        "fall_ref_phase": (None if fall_step is None else (
            "double" if (lib["foot_contact"][start + fall_step] > 0.5).all()
            else ("flight" if not (lib["foot_contact"][start + fall_step] > 0.5).any() else "single"))),
        "tilt_max_deg": round(float(np.max(metrics["tilt"])), 2),
        "success": bool(success),
        "entry_ok": bool(entry_ok),
        "entry_ok_old": bool(entry_ok_old),
        "entry_fall": bool(entry_fall),
        "entry_mismatch_post_pct": round(entry_mismatch_post_pct, 3),
        "entry_com_out_ds_frames": entry_com_out_ds,
        "entry_first_ss_frame": entry_first_ss,
        "entry_blend_end_frame": blend_end,
        "entry_first_lift_frame": entry_first_lift,
        "exit_ok": bool(exit_ok),
        "exit_drift_m": round(exit_drift, 4),
        "body_pos_err_rel_mean": round(float(body_err.mean()), 4),
        "body_pos_err_rel_p95": round(float(np.percentile(body_err, 95)), 4),
        "body_pos_err_abs_mean_m": round(float(np.mean(metrics["body_pos_err_abs"])), 4),
        "body_pos_err_abs_median_m": round(float(np.median(metrics["body_pos_err_abs"])), 4),
        "body_lin_vel_err_mean_mps": round(float(np.mean(metrics["body_lin_vel_err"])), 4),
        "joint_pos_err_mean": round(float(np.mean(metrics["joint_pos_err"])), 4),
        "anchor_pos_err_mean": round(float(np.mean(metrics["anchor_pos_err"])), 4),
        "contact_mismatch_pct": round(float(np.mean(metrics["contact_mismatch"]) / 2 * 100), 3),
        "slip_mean_mps": round(float(np.mean(metrics["slip"])), 4),
        "slip_stance_mean_mps": round(float(np.nanmean(metrics["slip_stance"])), 4),
        "slip_p95_mps": round(float(np.percentile(metrics["slip"], 95)), 4),
        "flight_ref_pct": round(float(np.mean(metrics["flight_ref"]) * 100), 2),
        "flight_robot_pct": round(float(np.mean(metrics["flight_robot"]) * 100), 2),
        "action_rate_mean": round(float(np.mean(metrics["action_rate"])), 4),
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
    if args.metrics:
        np.savez(args.metrics, **{k: np.array(v) for k, v in metrics.items() if not k.startswith("_")})
        print(f"[metrics] {args.metrics}")


if __name__ == "__main__":
    main()
