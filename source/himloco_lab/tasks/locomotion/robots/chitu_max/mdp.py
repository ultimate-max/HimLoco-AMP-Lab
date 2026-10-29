"""RobotLoad command sampling, payload randomization and task reward."""

import torch
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class ChituVelocityCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        straight = torch.rand(len(env_ids), device=self.device) < 0.1
        self.vel_command_b[env_ids[straight], 1:] = 0.0
        moving = self.vel_command_b[env_ids, :2].norm(dim=-1) > 0.05
        self.vel_command_b[env_ids, :2] *= moving.unsqueeze(-1)


@configclass
class ChituVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = ChituVelocityCommand


def randomize_payload(env, env_ids):
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies("payload")
    if len(body_ids) != 1:
        raise ValueError("ChituMax requires an unmerged payload link")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device="cpu")
    env_ids = env_ids.to(device="cpu", dtype=torch.long)
    body_id = body_ids[0]
    masses = robot.root_physx_view.get_masses()
    nominal = (env_ids % 2).float() * 10.0 + 5.0
    sampled = (nominal + torch.empty(len(env_ids)).uniform_(-5.0, 5.0)).clamp_min(0.01)
    inertias = robot.root_physx_view.get_inertias()
    inertias[env_ids, body_id] *= (sampled / masses[env_ids, body_id]).unsqueeze(-1)
    masses[env_ids, body_id] = sampled
    robot.root_physx_view.set_masses(masses, env_ids)
    robot.root_physx_view.set_inertias(inertias, env_ids)


def robotload_reward(env):
    """Return positive-clipped RobotLoad task reward as a reward rate.

    Individual terms are accumulated for the training log. Their sum is still
    clipped once, so the value returned to PPO is unchanged.
    """
    terms = _robotload_terms(env)
    if not hasattr(env, "_robotload_term_sums"):
        env._robotload_term_sums = {
            name: torch.zeros(env.num_envs, device=env.device) for name in terms
        }
    for name, value in terms.items():
        env._robotload_term_sums[name] += value
    reward = torch.stack(tuple(terms.values())).sum(dim=0)
    return reward.clamp_min(0.0) / env.step_dt


def _robotload_terms(env):
    robot = env.scene["robot"]
    data = robot.data
    command = env.command_manager.get_command("base_velocity")
    joint_pos = data.joint_pos[:, env._amp_joint_indices].reshape(-1, 4, 3)
    linear_vel = data.root_lin_vel_b
    angular_vel = data.root_ang_vel_b
    xy_command = command[:, :2].norm(dim=-1)
    stand = ((xy_command < 0.05) & (command[:, 2].abs() < 0.1)).to(linear_vel.dtype)
    yaw = ((xy_command < 0.05) & (command[:, 2].abs() > 0.1)).to(linear_vel.dtype)
    backward = (command[:, 0] < -0.05).to(linear_vel.dtype)
    moving_xy = (xy_command > 0.05).to(linear_vel.dtype)
    feet = data.body_pos_w[:, env._amp_foot_body_indices]
    foot_vel = data.body_lin_vel_w[:, env._amp_foot_body_indices]
    sensor = env.scene["contact_forces"]
    sensor_ids, _ = sensor.find_bodies(env.cfg.amp_foot_names, preserve_order=True)
    contact = sensor.data.net_forces_w[:, sensor_ids, 2] > 1.0
    height_error = (data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - 0.6).square()
    vertical_vel = linear_vel[:, 2].square()
    roll_pitch = angular_vel[:, :2].square().sum(-1)
    sigma = torch.where(backward.bool(), 0.15, 0.25)
    clearance = (feet[:, :, 2] - env.scene.env_origins[:, 2:3]).clamp_min(0.0)
    clearance = ((clearance / 0.080001).clamp(max=1.0) * ~contact).sum(-1)
    symmetry_sign = joint_pos.new_tensor([-1.0, 1.0, 1.0])
    symmetry = (joint_pos[:, :2] - joint_pos[:, 2:] * symmetry_sign).abs().sum((1, 2))
    limits = data.soft_joint_pos_limits
    position_excess = (limits[:, :, 0] - data.joint_pos).clamp_min(0.0)
    position_excess += (data.joint_pos - limits[:, :, 1]).clamp_min(0.0)
    slide = (foot_vel[:, :, :2].norm(dim=-1) * contact).sum(-1)
    force = robot.permanent_wrench_composer.composed_force_as_torch[:, :, :2].norm(dim=-1).amax(dim=-1)
    outward = joint_pos[:, :, 0] * joint_pos.new_tensor([1.0, 1.0, -1.0, -1.0])
    excess = (outward - 0.05).clamp_min(0.0)
    front, hind = excess[:, [0, 2]], excess[:, [1, 3]]
    abduction = (front > 0).all(-1) * front.square().sum(-1)
    abduction += (hind > 0).all(-1) * hind.square().sum(-1)
    abduction_weight = torch.exp(-(command[:, 1] / 0.25).square() - (command[:, 2] / 0.01).square())
    abduction_weight *= command[:, 0].abs() / (command.abs().sum(-1) + 0.001)
    return {
        "tracking_lin_vel": 15.5 * torch.exp(-(command[:, :2] - linear_vel[:, :2]).square().sum(-1) / sigma),
        "tracking_ang_vel": 10.75 * torch.exp(-(command[:, 2] - angular_vel[:, 2]).square() / 0.25),
        "lin_vel_z": -5.5 * vertical_vel,
        "lin_vel_z_yaw": -5.0 * yaw * vertical_vel,
        "ang_vel_xy": -6.45 * roll_pitch,
        "ang_vel_xy_stand": -0.75 * stand * roll_pitch,
        "orientation": -18.8 * data.projected_gravity_b[:, :2].square().sum(-1),
        "base_height": -266.6 * height_error,
        "base_height_yaw": -106.6 * yaw * height_error,
        "feet_clearance": 1.5 * moving_xy * clearance,
        "feet_clearance_bwd": 2.8 * backward * clearance,
        "stand_leg_symmetry": -10.0 * symmetry * stand,
        "dof_pos_limits": -0.2 * position_excess.sum(-1),
        "dof_vel_limits": -0.5 * (data.joint_vel.abs() - 6.3).clamp(0.0, 1.0).sum(-1),
        "feet_slide": -0.5 * slide,
        "feet_slide_stand": -0.5 * stand * slide,
        "lin_vel_xy_stand": -1.5 * stand * linear_vel[:, :2].square().sum(-1) * (1.0 + 2.0 * (force > 2.0)),
        "feet_contact_stand": -10.0 * stand * (~contact).sum(-1),
        "action_rate": -0.45 * (env.action_manager.action - env.action_manager.prev_action).square().sum(-1),
        "haa_abduction": -200.0 * abduction * abduction_weight,
    }
