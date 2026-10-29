#!/usr/bin/env python3
"""Export a ChituMax HimLoco checkpoint as a deploy-ready ONNX policy.

The ONNX matches rl_deploy_python/chitu_max/trot: one frame is 47-d, history is
oldest-first, and actions come back in policy joint order. The HIM network
itself consumes the 45-d training observation, newest frame first. That
conversion is inside the graph, as in AMP_for_hardware/scripts/export_policy_to_onnx.py.

Deploy frame (47), already scaled by trot.py:
  [0:3]   base angular velocity * 0.25
  [3:6]   projected gravity
  [6:9]   velocity command * 1.5
  [9:21]  joint position error, policy order FR FL RR RL
  [21:33] joint velocity * 0.05, policy order
  [33:45] previous action, policy order
  [45:47] gait phase, ignored

Training frame (45). Joints follow the Isaac articulation order, which is
all HAA, then all HFE, then all KFE (LF, LH, RF, RH within each group).
The position scale in that order is 0.18 four times, then 0.3 eight times:
  [0:3]   velocity command, unscaled
  [3:6]   base angular velocity * 0.25
  [6:9]   projected gravity
  [9:21]  joint position error
  [21:33] joint velocity * 0.05
  [33:45] previous action

The output is the raw action in policy order. trot.py then applies
0.18 / 0.3 / 0.3 per leg, which matches once the joints are back in
hip, thigh, calf order.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEPLOY_OBS_DIM = 47
TRAIN_OBS_DIM = 45
ONE_STEP_EXTRA = 3 + 16  # estimated linear velocity + latent
DEPLOY_CMD_SCALE = (1.5, 1.5, 1.5)
TRAIN_CMD_SCALE = (1.0, 1.0, 1.0)
# Isaac order: LF/LH/RF/RH HAA, then the same four legs HFE, then KFE.
# Policy order: FR FL RR RL, each leg hip, thigh, calf.
POLICY_TO_TRAIN = (3, 9, 0, 6, 4, 10, 1, 7, 5, 11, 2, 8)
TRAIN_TO_POLICY = (2, 6, 10, 0, 4, 8, 3, 7, 11, 1, 5, 9)
TRAIN_JOINT_NAMES = (
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
)
POLICY_JOINT_NAMES = (
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
)


def sequential_from_state_dict(state_dict, prefix):
    """Rebuild Linear+ELU blocks and copy checkpoint weights."""
    needle = prefix + "."
    weight_ids = sorted(
        int(key[len(needle):].split(".", 1)[0])
        for key in state_dict
        if key.startswith(needle) and key.endswith(".weight")
    )
    if not weight_ids:
        raise KeyError(f"No weights with prefix {prefix!r} in the checkpoint")
    layers = []
    for i, index in enumerate(weight_ids):
        weight = state_dict[f"{needle}{index}.weight"]
        bias = state_dict[f"{needle}{index}.bias"]
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        layers.append(layer)
        if i != len(weight_ids) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ChituPolicy(nn.Module):
    """Training observation history -> actions in URDF order."""

    def __init__(self, encoder, actor, one_step_obs):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.one_step_obs = int(one_step_obs)

    def forward(self, obs_history):
        parts = self.encoder(obs_history)
        vel = parts[..., :3]
        latent = F.normalize(parts[..., 3:], dim=-1, p=2)
        current = obs_history[..., : self.one_step_obs]
        return self.actor(torch.cat((current, vel, latent), dim=-1))


class DeployPolicyWrapper(nn.Module):
    """47-d deploy frames -> policy-order actions, matching chitu_max/trot.py."""

    def __init__(self, policy, history_steps):
        super().__init__()
        self.policy = policy
        self.history_steps = int(history_steps)
        self.register_buffer("policy_to_train_idx", torch.tensor(POLICY_TO_TRAIN, dtype=torch.long))
        self.register_buffer("train_to_policy_idx", torch.tensor(TRAIN_TO_POLICY, dtype=torch.long))
        rescale = torch.tensor(TRAIN_CMD_SCALE) / torch.tensor(DEPLOY_CMD_SCALE)
        self.register_buffer("cmd_rescale", rescale)

    def _frame_to_train(self, frame):
        command = frame[:, 6:9] * self.cmd_rescale
        angular_velocity = frame[:, 0:3]
        gravity = frame[:, 3:6]
        joint_pos = frame[:, 9:21][:, self.policy_to_train_idx]
        joint_vel = frame[:, 21:33][:, self.policy_to_train_idx]
        last_action = frame[:, 33:45][:, self.policy_to_train_idx]
        return torch.cat((command, angular_velocity, gravity, joint_pos, joint_vel, last_action), dim=-1)

    def forward(self, deploy_obs):
        frames = deploy_obs.view(-1, self.history_steps, DEPLOY_OBS_DIM)
        # trot.py stores oldest first; the estimator reads newest first.
        train_frames = [
            self._frame_to_train(frames[:, step, :])
            for step in range(self.history_steps - 1, -1, -1)
        ]
        actions = self.policy(torch.cat(train_frames, dim=-1))
        return actions[:, self.train_to_policy_idx]


def build_policy(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    encoder = sequential_from_state_dict(state_dict, "estimator.encoder")
    actor = sequential_from_state_dict(state_dict, "actor")
    obs_dim = encoder[0].in_features
    one_step_obs = actor[0].in_features - ONE_STEP_EXTRA
    if one_step_obs != TRAIN_OBS_DIM or obs_dim % one_step_obs != 0:
        raise ValueError(
            f"Expected a {TRAIN_OBS_DIM}-d Chitu observation, got encoder input {obs_dim}, "
            f"actor input {actor[0].in_features}"
        )
    policy = ChituPolicy(encoder, actor, one_step_obs)
    policy.eval()
    return policy, obs_dim // one_step_obs


def check_against_training_policy(wrapper, policy, history_steps):
    deploy = torch.randn(4, DEPLOY_OBS_DIM * history_steps)
    frames = deploy.view(4, history_steps, DEPLOY_OBS_DIM)
    train_frames = []
    rescale = torch.tensor(TRAIN_CMD_SCALE) / torch.tensor(DEPLOY_CMD_SCALE)
    policy_to_train = list(POLICY_TO_TRAIN)
    for step in range(history_steps - 1, -1, -1):
        frame = frames[:, step, :]
        train_frames.append(torch.cat((
            frame[:, 6:9] * rescale,
            frame[:, 0:3],
            frame[:, 3:6],
            frame[:, 9:21][:, policy_to_train],
            frame[:, 21:33][:, policy_to_train],
            frame[:, 33:45][:, policy_to_train],
        ), dim=-1))
    with torch.no_grad():
        reference = policy(torch.cat(train_frames, dim=-1))[:, list(TRAIN_TO_POLICY)]
        exported = wrapper(deploy)
    mismatch = (reference - exported).abs().max().item()
    if mismatch > 1e-6:
        raise RuntimeError(f"Deploy wrapper does not match the training policy: max abs {mismatch}")
    for train_index, name in enumerate(TRAIN_JOINT_NAMES):
        if POLICY_TO_TRAIN[train_index] != POLICY_JOINT_NAMES.index(name):
            raise RuntimeError(f"{name} is not gathered from the policy joint of the same name")
    for policy_index, name in enumerate(POLICY_JOINT_NAMES):
        if TRAIN_TO_POLICY[policy_index] != TRAIN_JOINT_NAMES.index(name):
            raise RuntimeError(f"{name} is not scattered back to the policy joint of the same name")


def export_onnx(checkpoint_path, output_path, opset_version=11):
    print(f"加载 checkpoint: {checkpoint_path}")
    policy, history_steps = build_policy(checkpoint_path)
    wrapper = DeployPolicyWrapper(policy, history_steps)
    wrapper.eval()
    deploy_dim = DEPLOY_OBS_DIM * history_steps
    print(f"  训练单帧: {TRAIN_OBS_DIM}，历史 {history_steps} 帧（网络内最新帧在前）")
    print(f"  ONNX 输入: obs ({deploy_dim})，与 chitu_max/trot 的 47 维单帧一致")
    print(f"  指令缩放: 部署 {DEPLOY_CMD_SCALE} -> 训练 {TRAIN_CMD_SCALE}")
    print("  输出: actions (12)，policy 序 FR/FL/RR/RL，未乘动作缩放")
    print("  部署配置保持 obs_dim: 47, obs_history_frames: "
          f"{history_steps}, command_scale: [1.5, 1.5, 1.5]")

    check_against_training_policy(wrapper, policy, history_steps)

    dummy = torch.zeros(1, deploy_dim)
    with torch.no_grad():
        output = wrapper(dummy)
    if output.shape != (1, 12):
        raise RuntimeError(f"Unexpected output shape {tuple(output.shape)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
        opset_version=opset_version,
    )
    print(f"\nONNX 模型已导出到: {output_path}")

    import onnx

    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX 模型验证通过")
    return wrapper, deploy_dim


def verify_with_onnxruntime(wrapper, onnx_path, obs_dim, num_tests=8):
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nonnxruntime 未安装，跳过运行时验证")
        return

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    max_diff = 0.0
    with torch.no_grad():
        for _ in range(num_tests):
            obs = torch.randn(1, obs_dim)
            reference = wrapper(obs).numpy()
            exported = session.run(None, {input_name: obs.numpy()})[0]
            max_diff = max(max_diff, float(np.max(np.abs(reference - exported))))
    print(f"\nonnxruntime 对比 {num_tests} 组随机输入，最大绝对误差: {max_diff:.3e}")
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX 输出与 PyTorch 不一致: max abs {max_diff}")


def main():
    parser = argparse.ArgumentParser(description="将 ChituMax HimLoco checkpoint 导出为可直接部署的 ONNX")
    parser.add_argument("--checkpoint", required=True, help="训练保存的 model_*.pt")
    parser.add_argument(
        "--output",
        default=None,
        help="ONNX 路径。默认写到 checkpoint 同级目录的 exported/chitu_max_policy.onnx",
    )
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(args.checkpoint), "exported", "chitu_max_policy.onnx")
    wrapper, obs_dim = export_onnx(args.checkpoint, output, args.opset)
    verify_with_onnxruntime(wrapper, output, obs_dim)


if __name__ == "__main__":
    main()
