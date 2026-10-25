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
    nominal = (env_ids % 4).float() * 10.0 + 5.0
    sampled = (nominal + torch.empty(len(env_ids)).uniform_(-5.0, 5.0)).clamp_min(0.01)
    inertias = robot.root_physx_view.get_inertias()
    inertias[env_ids, body_id] *= (sampled / masses[env_ids, body_id]).unsqueeze(-1)
    masses[env_ids, body_id] = sampled
    robot.root_physx_view.set_masses(masses, env_ids)
    robot.root_physx_view.set_inertias(inertias, env_ids)


def robotload_reward(env):
    """Return positive-clipped RobotLoad task reward as a reward rate."""
    robot = env.scene["robot"]
    data = robot.data
    command = env.command_manager.get_command("base_velocity")
    joint_pos = data.joint_pos[:, env._amp_joint_indices].reshape(-1, 4, 3)
    linear_vel = data.root_lin_vel_b
    angular_vel = data.root_ang_vel_b
    xy_command = command[:, :2].norm(dim=-1)
    stand = (xy_command < 0.05) & (command[:, 2].abs() < 0.1)
    yaw = (xy_command < 0.05) & (command[:, 2].abs() > 0.1)
    backward = command[:, 0] < -0.05
    feet = data.body_pos_w[:, env._amp_foot_body_indices]
    foot_vel = data.body_lin_vel_w[:, env._amp_foot_body_indices]
    sensor = env.scene["contact_forces"]
    sensor_ids, _ = sensor.find_bodies(env.cfg.amp_foot_names, preserve_order=True)
    contact = sensor.data.net_forces_w[:, sensor_ids, 2] > 1.0
    height_error = (data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - 0.6).square()
    sigma = torch.where(backward, 0.18, 0.35)
    reward = 15.5 * torch.exp(-(command[:, :2] - linear_vel[:, :2]).square().sum(-1) / sigma)
    reward += 10.75 * torch.exp(-(command[:, 2] - angular_vel[:, 2]).square() / 0.35)
    reward -= (5.5 + 5.0 * yaw) * linear_vel[:, 2].square()
    reward -= (18.45 + 0.75 * stand) * angular_vel[:, :2].square().sum(-1)
    reward -= 18.8 * data.projected_gravity_b[:, :2].square().sum(-1)
    reward -= (1666.6 + 106.6 * yaw) * height_error
    clearance = (feet[:, :, 2] - env.scene.env_origins[:, 2:3]).clamp_min(0.0)
    clearance = ((clearance / 0.080001).clamp(max=1.0) * ~contact).sum(-1)
    reward += (0.3 * (xy_command > 0.05) + 1.5 * backward) * clearance
    symmetry_sign = joint_pos.new_tensor([-1.0, 1.0, 1.0])
    symmetry = (joint_pos[:, :2] - joint_pos[:, 2:] * symmetry_sign).abs().sum((1, 2))
    reward -= 10.0 * symmetry * stand
    limits = data.soft_joint_pos_limits
    position_excess = (limits[:, :, 0] - data.joint_pos).clamp_min(0.0)
    position_excess += (data.joint_pos - limits[:, :, 1]).clamp_min(0.0)
    reward -= 0.2 * position_excess.sum(-1)
    reward -= 0.5 * (data.joint_vel.abs() - 6.3).clamp(0.0, 1.0).sum(-1)
    slide = (foot_vel[:, :, :2].norm(dim=-1) * contact).sum(-1)
    reward -= (0.5 + 0.5 * stand) * slide
    force = robot.permanent_wrench_composer.composed_force_as_torch[:, :, :2].norm(dim=-1).amax(dim=-1)
    reward -= 1.5 * stand * linear_vel[:, :2].square().sum(-1) * (1.0 + 2.0 * (force > 2.0))
    reward -= 10.0 * stand * (~contact).sum(-1)
    reward -= 0.4 * (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)
    outward = joint_pos[:, :, 0] * joint_pos.new_tensor([1.0, 1.0, -1.0, -1.0])
    excess = (outward - 0.05).clamp_min(0.0)
    front, hind = excess[:, [0, 2]], excess[:, [1, 3]]
    abduction = (front > 0).all(-1) * front.square().sum(-1)
    abduction += (hind > 0).all(-1) * hind.square().sum(-1)
    weight = torch.exp(-(command[:, 1] / 0.25).square() - (command[:, 2] / 0.01).square())
    weight *= command[:, 0].abs() / (command.abs().sum(-1) + 0.001)
    reward -= 200.0 * abduction * weight
    return reward.clamp_min(0.0) / env.step_dt
