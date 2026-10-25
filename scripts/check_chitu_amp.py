"""CPU preflight for bundled RobotLoad assets and an AMP/PPO training update.

Run with a Python environment containing NumPy and PyTorch; Isaac Sim is not needed.
"""

import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def main():
    torch.manual_seed(7)
    np.random.seed(7)
    torch.set_num_threads(2)
    package("_chitu_check", ROOT / "source/himloco_lab")
    package("_chitu_check.rsl_rl", ROOT / "source/himloco_lab/rsl_rl")
    data = importlib.import_module("_chitu_check.assets.chitu_data")
    loader_type = importlib.import_module("_chitu_check.rsl_rl.datasets.motion_loader").AMPLoader
    foot_positions = importlib.import_module("_chitu_check.rsl_rl.datasets.kinematics").foot_positions
    symmetry = importlib.import_module("_chitu_check.rsl_rl.utils.chitu_symmetry")
    urdf_path = data.ROBOTLOAD_ROOT / "resources/robots/chitu/urdf/chitu.urdf"
    robot = ET.parse(urdf_path).getroot()
    assert robot.attrib["name"] == "chitu_max"
    joints = {joint.attrib["name"]: joint for joint in robot.findall("joint")}
    for name in data.CHITU_JOINT_NAMES:
        assert name in joints and joints[name].attrib["type"] == "revolute"
    for mesh in robot.findall(".//mesh"):
        assert (urdf_path.parent / mesh.attrib["filename"]).is_file()
    assert "payload" in {link.attrib["name"] for link in robot.findall("link")}
    assert len(data.CHITU_MOTION_FILES) > 0
    with contextlib.redirect_stdout(io.StringIO()):
        loader = loader_type(
            "cpu", 0.02, motion_files=data.CHITU_MOTION_FILES,
            joint_reorder="chitu", foot_fk_params=data.CHITU_FK_PARAMS,
            recompute_joint_vel=True, preload_transitions=True, num_preload_transitions=128,
        )
    assert loader.observation_dim == 43
    for path, frames in zip(data.CHITU_MOTION_FILES, loader.trajectories_full):
        raw = json.loads(Path(path).read_text())
        raw_frames = np.asarray(raw["Frames"], dtype=np.float32)
        expected = raw_frames[:, 7:19].reshape(-1, 4, 3)[:, [1, 3, 0, 2]].reshape(-1, 12)
        np.testing.assert_allclose(frames[:, 7:19], expected)
        np.testing.assert_allclose(frames[:, 37:49], np.gradient(expected, raw["FrameDuration"], axis=0))
        assert torch.isfinite(frames).all()
    sample = loader.get_full_frame_batch(32)
    torch.testing.assert_close(sample[:, 19:31], foot_positions(sample[:, 7:19], data.CHITU_FK_PARAMS),
                               atol=0.002, rtol=0.002)
    angles = torch.randn(8, 12) * 0.2
    expected_feet = []
    for leg in range(4):
        hip, thigh, calf = angles[:, leg * 3:leg * 3 + 3].unbind(-1)
        def rotation(angle, axis):
            matrix = torch.eye(3).repeat(len(angle), 1, 1)
            first, second = (1, 2) if axis == "x" else (2, 0)
            matrix[:, first, first] = matrix[:, second, second] = angle.cos()
            matrix[:, first, second] = -angle.sin()
            matrix[:, second, first] = angle.sin()
            return matrix
        hip_rot, thigh_rot, calf_rot = rotation(hip, "x"), rotation(thigh, "y"), rotation(calf, "y")
        offsets = {key: torch.tensor(value[leg]) for key, value in data.CHITU_FK_PARAMS.items()}
        expected_feet.append(offsets["hip_offsets"] + hip_rot @ offsets["hfe_offsets"] +
                             (hip_rot @ thigh_rot) @ offsets["kfe_offsets"] +
                             (hip_rot @ thigh_rot @ calf_rot) @ offsets["foot_offsets"])
    torch.testing.assert_close(foot_positions(angles, data.CHITU_FK_PARAMS), torch.cat(expected_feet, dim=-1))
    native_names = [f"{leg}_{joint}_joint" for joint in ("HAA", "HFE", "KFE") for leg in ("LF", "LH", "RF", "RH")]
    ops = symmetry.build_symmetry(native_names, 45, "cpu")
    observations = torch.randn(16, 45 * 6)
    torch.testing.assert_close(symmetry.mirror_observations(symmetry.mirror_observations(observations, ops), ops),
                               observations)
    gms = importlib.import_module("_chitu_check.assets.gms_friction")
    friction = gms.GmsFriction(8, 12, gms.load_gms_json(
        str(data.ROBOTLOAD_ROOT / "resources/friction/chitu_mgms_net_params.json")), "cpu", 0.005)
    assert torch.isfinite(friction.step(angles, torch.randn_like(angles), torch.ones_like(angles))).all()
    friction.reset(torch.arange(8))
    assert torch.count_nonzero(friction.delta) == 0
    actor_type = importlib.import_module("_chitu_check.rsl_rl.modules.amp_actor_critic").AMPActorCritic
    discriminator_type = importlib.import_module("_chitu_check.rsl_rl.algorithms.amp_discriminator").AMPDiscriminator
    algorithm_type = importlib.import_module("_chitu_check.rsl_rl.algorithms.amp_ppo").AMPPPO
    normalizer_type = importlib.import_module("_chitu_check.rsl_rl.utils.normalizer").Normalizer
    with contextlib.redirect_stdout(io.StringIO()):
        actor = actor_type(45, 51, 45, 12, actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16])
        discriminator = discriminator_type(86, 10.0, [32, 16], "cpu", 0.28)
    algorithm = algorithm_type(
        actor, discriminator, loader, normalizer_type(43),
        num_learning_epochs=1, num_mini_batches=2, amp_replay_buffer_size=128,
        symmetry_coef=15.0, min_std=torch.full((12,), 0.05),
    )
    algorithm.symmetry_ops = ops
    algorithm.init_storage(16, 4, [45], [51], [12])
    initial_actor = next(actor.actor.parameters()).detach().clone()
    initial_discriminator = next(discriminator.parameters()).detach().clone()
    for step in range(4):
        with torch.inference_mode():
            state, next_state = next(loader.feed_forward_generator(1, 16))
            algorithm.act(torch.randn(16, 45), torch.randn(16, 51), state)
            rewards, _ = discriminator.predict_amp_reward(state, next_state, torch.ones(16),
                                                          algorithm.amp_normalizer)
            algorithm.process_env_step(rewards, torch.zeros(16), {}, torch.randn(16, 51), next_state)
    with torch.inference_mode():
        algorithm.compute_returns(torch.randn(16, 51))
    losses = algorithm.update()
    assert np.isfinite(losses).all()
    assert not torch.equal(initial_actor, next(actor.actor.parameters()))
    assert not torch.equal(initial_discriminator, next(discriminator.parameters()))
    print(f"PASS: {len(data.CHITU_MOTION_FILES)} motions; assets; joint order; velocity gradients; FK; symmetry; MGMS; AMP/PPO update")
    print("Isaac Lab physics, task registration and convergence still require a simulator smoke run.")


if __name__ == "__main__":
    main()
