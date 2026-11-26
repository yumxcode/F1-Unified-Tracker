from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def flight_contact_penalty(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, std: float = 20.0
) -> torch.Tensor:
    """Reward for zero foot contact force while the reference is airborne.

    CLF-RL's flight-phase contact penalty, expressed as a positive reward:
    r = exp(-sum(F_feet) / std) on flight frames, 1 on stance frames.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    flight = command.contact_phase[:, 2]
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    f = torch.norm(forces, dim=-1).sum(dim=-1)
    return torch.exp(-(flight * f) / std)


def stance_foot_velocity(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.5
) -> torch.Tensor:
    """Penalise horizontal speed of a foot whose reference is in stance."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    phase = command.contact_phase[:, :2]
    idx = [command.cfg.body_names.index(n) for n in ("left_ankle_roll", "right_ankle_roll")]
    vel = command.robot_body_lin_vel_w[:, idx, :2]
    speed = torch.norm(vel, dim=-1)
    return torch.exp(-(phase * speed).sum(dim=-1) / std)


def stance_foot_height(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.05
) -> torch.Tensor:
    """Penalise a reference-stance foot whose ankle is still high.

    The contact-phase reward only checks forces; a foot can be "in contact"
    with the sole 3-5 cm above the floor (floating reference / soft PD).  This
    term plants the stance foot by penalising ankle z above 6 cm.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    phase = command.contact_phase[:, :2]
    idx = [command.cfg.body_names.index(n) for n in ("left_ankle_roll", "right_ankle_roll")]
    z = command.robot_body_pos_w[:, idx, 2]
    excess = torch.clamp(z - 0.06, min=0.0)
    return torch.exp(-(phase * excess).sum(dim=-1) / std)


def swing_clearance(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.05
) -> torch.Tensor:
    """Penalise a low swing foot (the B6 "drag" failure mode).

    When the reference is in swing, the robot's ankle should be above ~5 cm;
    a foot at 2-3 cm drags on the floor and causes contact mismatch.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    swing = 1.0 - command.contact_phase[:, :2]
    idx = [command.cfg.body_names.index(n) for n in ("left_ankle_roll", "right_ankle_roll")]
    z = command.robot_body_pos_w[:, idx, 2]
    deficit = torch.clamp(0.05 - z, min=0.0) * swing
    return torch.exp(-deficit.sum(dim=-1) / std)


def flight_mismatch(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, std: float = 0.5
) -> torch.Tensor:
    """Match the reference aerial phase (binary) with the robot's foot forces."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_flight = command.contact_phase[:, 2]
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    f = torch.norm(forces, dim=-1).sum(dim=-1)
    robot_flight = (f < 1.0).float()
    return torch.exp(-torch.abs(robot_flight - ref_flight) / std)


def foot_slippage(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.3
) -> torch.Tensor:
    """Same as stance_foot_velocity but normalised per foot and sharper."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    phase = command.contact_phase[:, :2]
    idx = [command.cfg.body_names.index(n) for n in ("left_ankle_roll", "right_ankle_roll")]
    vel = command.robot_body_lin_vel_w[:, idx, :2]
    speed = torch.norm(vel, dim=-1)
    return torch.exp(-(phase * speed).mean(dim=-1) / std)
