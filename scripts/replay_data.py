"""Kinematically replay AMP clips in Isaac Sim."""

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
parser.add_argument(
    "--motion",
    default=None,
    help="Clip to replay: filename stem (yaw_0p5), filename, path, or zero-based index. "
    "Default replays every configured clip.",
)
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


def resolve_motion_files(configured: list[str], selection: str | None) -> list[str]:
    """Pick one configured clip by stem, filename, path, or index."""
    files = [Path(path) for path in configured]
    if not files:
        raise ValueError("AMP motion_files is empty")
    if selection is None:
        return [str(path) for path in files]
    if selection.isdigit():
        index = int(selection)
        if not 0 <= index < len(files):
            raise ValueError(f"--motion index must be in [0, {len(files) - 1}]")
        return [str(files[index])]
    direct = Path(selection).expanduser()
    if direct.is_file():
        return [str(direct)]
    matches = [path for path in files if selection in (path.name, path.stem)]
    if len(matches) == 1:
        return [str(matches[0])]
    available = ", ".join(path.stem for path in files)
    if not matches:
        raise ValueError(f"No motion named {selection!r}. Available: {available}")
    raise ValueError(f"--motion {selection!r} matches more than one file: {available}")


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

    motion_files = resolve_motion_files(list(agent_cfg.amp_motion_files), args_cli.motion)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene.articulations["robot"]
    amp = AMPLoader(
        device=base_env.device,
        time_between_frames=base_env.step_dt,
        motion_files=motion_files,
        reorder=getattr(agent_cfg, "reorder_motion_data", False),
        joint_reorder=getattr(agent_cfg, "amp_joint_reorder", None),
        foot_fk_params=getattr(agent_cfg, "amp_foot_fk_params", None),
        recompute_joint_vel=getattr(agent_cfg, "amp_recompute_joint_vel", False),
    )

    joint_ids, joint_names = robot.find_joints(env_cfg.amp_joint_names, preserve_order=True)
    foot_ids, foot_names = robot.find_bodies(env_cfg.amp_foot_names, preserve_order=True)
    if joint_names != list(env_cfg.amp_joint_names) or foot_names != list(env_cfg.amp_foot_names):
        raise RuntimeError("Asset names do not match canonical AMP ordering")

    env.reset()
    zero_gains = torch.zeros(1, robot.num_joints, device=base_env.device)
    robot.write_joint_stiffness_to_sim(zero_gains)
    robot.write_joint_damping_to_sim(zero_gains)
    print(f"\n[{args_cli.task} REPLAY] motions={amp.num_motions}")
    for path in motion_files:
        print(f"  {path}")
    print(f"joints={joint_names}")
    print(f"feet={foot_names}\n")

    origin = base_env.scene.env_origins[:1]
    step_count = error_count = 0
    error_sum_sq = max_error = 0.0

    while simulation_app.is_running():
        for traj_idx, path in enumerate(motion_files):
            anchor_xy = AMPLoader.get_root_pos(amp.trajectories_full[traj_idx][0])[:2]
            print(f"[PLAY] {Path(path).name}")
            t = 0.0
            while t < amp.trajectory_lens[traj_idx] and simulation_app.is_running():
                start = time.perf_counter()
                frame = amp.get_full_frame_at_time_batch(np.array([traj_idx]), np.array([t]))
                root_xyzw = amp.get_root_rot_batch(frame)
                root_wxyz = torch.cat((root_xyzw[:, 3:4], root_xyzw[:, :3]), dim=-1)
                root_pos = amp.get_root_pos_batch(frame).clone()
                root_pos[:, :2] += origin[:, :2] - anchor_xy
                root_pos[:, 2] += origin[:, 2]
                robot.write_root_pose_to_sim(torch.cat((root_pos, root_wxyz), dim=-1))
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

                if not args_cli.headless:
                    base_env.sim.render()
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
