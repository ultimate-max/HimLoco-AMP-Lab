from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels( env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str = "track_lin_vel_xy",) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity") # 获取速度命令项
    ranges = command_term.cfg.ranges # 获取速度命令项的范围配置
    limit_ranges = command_term.cfg.curriculums_limit_ranges # 获取速度命令项的课程限制范围配置
    
    # 根据环境ID将环境分为低速度组和高速度组
    low_vel_env_ids = (env_ids >= (env.num_envs * command_term.cfg.rel_high_vel_envs))
    high_vel_env_ids = (env_ids < (env.num_envs * command_term.cfg.rel_high_vel_envs))
    low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
    high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

    # 获取奖励项配置
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    # 只有在每个环境的一个episode结束时才更新课程
    if env.common_step_counter % env.max_episode_length == 0:
        # 计算低速度组和高速度组的平均奖励
        reward_low = torch.mean(env.reward_manager._episode_sums[reward_term_name][low_vel_env_ids]) / env.max_episode_length_s if len(low_vel_env_ids) > 0 else 0.0
        reward_high = torch.mean(env.reward_manager._episode_sums[reward_term_name][high_vel_env_ids]) / env.max_episode_length_s if len(high_vel_env_ids) > 0 else 0.0

        # 调试打印
        threshold = reward_term.weight * 0.8
        print(f"[Curriculum] step={env.common_step_counter}, "
              f"low_vel_envs={len(low_vel_env_ids)}, high_vel_envs={len(high_vel_env_ids)}, "
              f"reward_low={reward_low:.4f}, reward_high={reward_high:.4f}, "
              f"threshold={threshold:.4f}, "
              f"low_ok={reward_low > threshold}, high_ok={reward_high > threshold}, "
              f"lin_vel_x_range={ranges.lin_vel_x}")

        # Only update curriculum if both groups are performing well
        if reward_low > reward_term.weight * 0.8 and reward_high > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.2, 0.2], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges[0],
                limit_ranges[1],
            ).tolist()
            print(f"[Curriculum] ✅ UPGRADED! New lin_vel_x range: {ranges.lin_vel_x}")

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)
