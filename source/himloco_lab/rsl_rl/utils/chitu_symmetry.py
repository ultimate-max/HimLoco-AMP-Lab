"""RobotLoad left/right Gaussian policy symmetry in HimLoco observation order."""

import torch


def build_symmetry(joint_names, frame_dim, device):
    if len(joint_names) != 12 or frame_dim != 45:
        raise ValueError("Chitu symmetry expects 12 joints and 45 policy observations")
    partners = {"LF": "RF", "RF": "LF", "LH": "RH", "RH": "LH"}
    mirrored_names = [partners[name[:2]] + name[2:] for name in joint_names]
    action_perm = torch.tensor([joint_names.index(name) for name in mirrored_names], device=device)
    action_sign = torch.tensor([-1.0 if "_HAA_" in name else 1.0 for name in joint_names], device=device)
    obs_perm = torch.arange(frame_dim, device=device)
    obs_sign = torch.ones(frame_dim, device=device)
    obs_sign[:9] = obs_sign.new_tensor([1, -1, -1, -1, 1, -1, 1, -1, 1])
    for offset in (9, 21, 33):
        obs_perm[offset:offset + 12] = action_perm + offset
        obs_sign[offset:offset + 12] = action_sign
    return obs_perm, obs_sign, action_perm, action_sign


def mirror_observations(observations, ops):
    obs_perm, obs_sign = ops[:2]
    frames = observations.reshape(observations.shape[0], -1, len(obs_perm))
    return (frames[:, :, obs_perm] * obs_sign).reshape_as(observations)


def gaussian_kl(mean, std, other_mean, other_std):
    return (torch.log(other_std / std) +
            (std.square() + (mean - other_mean).square()) / (2 * other_std.square()) - 0.5).sum(-1)
