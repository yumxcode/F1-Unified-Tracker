import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("GMR/assets/xyber_x1/xyber_x1_mocap.xml")
d = mujoco.MjData(m)
# home keyframe was stripped from the mocap xml; set the home pose manually
home = {
    "left_hip_pitch_joint": 0.4889, "left_hip_roll_joint": 0.0621, "left_hip_yaw_joint": -0.3385,
    "left_knee_pitch_joint": 0.632, "left_ankle_pitch_joint": -0.2722,
    "right_hip_pitch_joint": -0.4889, "right_hip_roll_joint": -0.0621, "right_hip_yaw_joint": 0.3385,
    "right_knee_pitch_joint": 0.632, "right_ankle_pitch_joint": -0.2722,
}
mujoco.mj_resetData(m, d)
for name, val in home.items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    d.qpos[m.jnt_qposadr[jid]] = val
mujoco.mj_forward(m, d)

for side in ("left", "right"):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
    R = d.xmat[bid].reshape(3, 3)
    # foot corner spheres local (x, y, z)
    corners = np.array([[0.03, -0.0408 if side == "left" else 0.0408, 0.07],
                        [-0.03, -0.0408 if side == "left" else 0.0408, 0.07],
                        [0.03, -0.0408 if side == "left" else 0.0408, -0.07],
                        [-0.03, -0.0408 if side == "left" else 0.0408, -0.07]])
    world = (d.xpos[bid][None, :] + corners @ R.T)
    print(f"{side}: ankle z={d.xpos[bid][2]:.4f}  corners z={np.round(world[:,2],4)}")
    # express world-up in the foot local frame at home pose
    up_local = R.T @ np.array([0, 0, 1.0])
    print(f"  world-up in foot local frame: {np.round(up_local, 4)}")
    print(f"  ankle height above lowest corner: {d.xpos[bid][2] - world[:,2].min():.4f}")
