"""Basic locomotion rewards used by the Go2 configuration."""
from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from himloco_lab.envs import HimlocoManagerBasedRLEnv

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚能耗"""
    asset: Articulation = env.scene[asset_cfg.name] 

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids] # 关节速度
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids] # 关节力矩
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def smoothness(env: HimlocoManagerBasedRLEnv) -> torch.Tensor:
    """惩罚动作变化率，使用二阶差分。"""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action*2 + env.pre_pre_action), dim=1)

def feet_height_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    command_name: str|None = None,
) -> torch.Tensor:
    """惩罚足端高度偏离目标高度，使用L2核函数，并将位置转换到身体坐标系下计算。零命令时不计算奖励。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(env.num_envs, -1)
    reward = torch.sum(foot_z_target_error * foot_leteral_vel, dim=1)
    if command_name is not None:
        reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward

def base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """奖励机器人保持在指定高度。

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None: # 如果提供了传感器配置，使用传感器数据调整目标高度以适应地形
        sensor: RayCaster = env.scene[sensor_cfg.name]
        
        ray_hits = sensor.data.ray_hits_w[..., 2]
        
        # 过滤无效值（NaN、Inf、过大值），并将它们替换为NaN以便后续计算忽略
        valid_mask = ~torch.isnan(ray_hits) & ~torch.isinf(ray_hits) & (torch.abs(ray_hits) < 1e6) # 根据实际情况调整阈值
        ray_hits_masked = torch.where(valid_mask, ray_hits, torch.tensor(float('nan'), device=ray_hits.device)) # 将无效值替换为NaN，以便后续计算中自动忽略这些值
        
        # 调整目标高度：当前目标高度 + 有效传感器读数的平均值（如果所有读数无效，则保持原目标高度不变）
        # 对有效读数取平均，忽略无效值（NaN）
        adjusted_heights = torch.nanmean(ray_hits_masked, dim=1)
        
        # For environments where ALL points are invalid, nanmean returns NaN
        # Replace these with current robot height
        all_invalid_mask = torch.isnan(adjusted_heights)
        if all_invalid_mask.any():
            invalid_env_count = all_invalid_mask.sum().item()
            total_invalid_points = (~valid_mask).sum().item()
            print(f"[WARNING] base_height - {invalid_env_count} envs with all invalid points, total invalid: {total_invalid_points}/{ray_hits.numel()} ({total_invalid_points/ray_hits.numel():.2%})")
            adjusted_heights[all_invalid_mask] = asset.data.root_link_pos_w[all_invalid_mask, 2] - target_height
        
        adjusted_target_height = target_height + adjusted_heights
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
