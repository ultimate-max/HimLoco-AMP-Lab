"""Kinematically replay Go2 AMP clips in Isaac Sim."""

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher
from himloco_rsl_rl import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Replay configured AMP trajectories.")
parser.add_argument("--task", default="Unitree-Go2-AMP-Play")
parser.add_argument("--agent", default="himloco_rsl_rl_cfg")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--motion", type=int, default=None, help="Only replay this zero-based clip index.")
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--real_time", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--seed", type=int, default=None)
cli_args.add_himloco_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_tasks.utils.hydra import hydra_task_config

import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from himloco_lab.rsl_rl.datasets.motion_loader import AMPLoader


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    for name in ("physics_material", "add_base_mass", "randomize_rigid_body_com", "push_robot", "external_force"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    for name in ("time_out", "base_contact", "terrain_out_of_bounds"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.critic.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene.articulations["robot"]
    amp = AMPLoader(
        device=base_env.device,
        time_between_frames=base_env.step_dt,
        motion_files=list(agent_cfg.amp_motion_files),
        reorder=agent_cfg.reorder_motion_data,
    )

    joint_ids, joint_names = robot.find_joints(env_cfg.amp_joint_names, preserve_order=True)
    foot_ids, foot_names = robot.find_bodies(env_cfg.amp_foot_names, preserve_order=True)
    if joint_names != list(env_cfg.amp_joint_names) or foot_names != list(env_cfg.amp_foot_names):
        raise RuntimeError("Asset names do not match canonical AMP ordering")

    env.reset()
    zero_gains = torch.zeros(1, robot.num_joints, device=base_env.device)
    robot.write_joint_stiffness_to_sim(zero_gains)
    robot.write_joint_damping_to_sim(zero_gains)
    print(f"\n[Go2 AMP REPLAY] motions={amp.num_motions}")
    print(f"joints={joint_names}")
    print(f"feet={foot_names}\n")

    if args_cli.motion is not None and not 0 <= args_cli.motion < amp.num_motions:
        raise ValueError(f"--motion must be in [0, {amp.num_motions - 1}]")
    indices = [args_cli.motion] if args_cli.motion is not None else list(range(amp.num_motions))
    actions = torch.zeros(1, base_env.action_manager.total_action_dim, device=base_env.device)
    step_count = error_count = 0
    error_sum_sq = max_error = 0.0

    while simulation_app.is_running():
        for traj_idx in indices:
            print(f"[PLAY] {traj_idx}: {Path(amp.trajectory_names[traj_idx]).name}")
            t = 0.0
            while t < amp.trajectory_lens[traj_idx] and simulation_app.is_running():
                start = time.perf_counter()
                frame = amp.get_full_frame_at_time_batch(np.array([traj_idx]), np.array([t]))
                root_xyzw = amp.get_root_rot_batch(frame)
                root_wxyz = torch.cat((root_xyzw[:, 3:4], root_xyzw[:, :3]), dim=-1)
                robot.write_root_pose_to_sim(torch.cat((amp.get_root_pos_batch(frame), root_wxyz), dim=-1))
                robot.write_root_velocity_to_sim(torch.cat((
                    quat_apply(root_wxyz, amp.get_linear_vel_batch(frame)),
                    quat_apply(root_wxyz, amp.get_angular_vel_batch(frame))), dim=-1))
                robot.write_joint_state_to_sim(
                    amp.get_joint_pose_batch(frame), amp.get_joint_vel_batch(frame), joint_ids=joint_ids)
                base_env.sim.forward()
                base_env.scene.update(dt=0.0)

                foot_w = robot.data.body_pos_w[:, foot_ids]
                root_w = robot.data.root_pos_w[:, None, :]
                quat_w = robot.data.root_quat_w[:, None, :].expand(-1, len(foot_ids), -1)
                foot_local = quat_apply_inverse(
                    quat_w.reshape(-1, 4), (foot_w - root_w).reshape(-1, 3)).reshape(1, -1)
                errors = (foot_local - amp.get_tar_toe_pos_local_batch(frame)).abs()
                error_sum_sq += float(torch.square(errors).sum())
                error_count += errors.numel()
                max_error = max(max_error, float(errors.max()))

                env.step(actions)
                step_count += 1
                t += base_env.step_dt
                if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
                    break
                if args_cli.real_time:
                    time.sleep(max(0.0, base_env.step_dt - (time.perf_counter() - start)))
            if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
                break
        if args_cli.max_steps is not None:
            break

    rmse = (error_sum_sq / max(error_count, 1)) ** 0.5
    print(f"\n[VERIFY] frames={step_count}, foot RMSE={rmse:.6f} m, max_abs={max_error:.6f} m")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
