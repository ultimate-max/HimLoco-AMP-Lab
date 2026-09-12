# HimLoco-AMP-Lab

[中文文档](README_CN.md)

An Isaac Lab project for Unitree Go2 locomotion using PPO, history-based
observations, and Adversarial Motion Priors (AMP). It includes training,
playback/export, expert-motion replay, and MuJoCo sim-to-sim tools.

## Features

- AMP and pure PPO velocity-control tasks for Unitree Go2.
- Terrain curriculum, domain randomization, and task rewards.
- Six-frame actor observations and an AMP discriminator.
- Kinematic replay for locally supplied expert trajectories.
- TorchScript/ONNX export and MuJoCo sim-to-sim.
- A publication check for private data and generated artifacts.

## Installation

Install Isaac Sim and Isaac Lab first. The current setup targets the Isaac Sim
5.1 / Python 3.11 environment used during development; other versions may need
adaptation. Activate that environment and run:

```bash
git clone <repository-url>
cd HimLoco-AMP-Lab
python -m pip install -e source
python scripts/list_envs.py
```

The Go2 description is bundled in `urdf/go2/`; keep it beside `source/`.
Its upstream license is retained in `urdf/go2/LICENSE`.

## Tasks and commands

| Task | Purpose |
| --- | --- |
| `Unitree-Go2-AMP` | AMP training |
| `Unitree-Go2-AMP-Play` | AMP playback and export |
| `Unitree-Go2-Velocity` | Pure PPO training |
| `Unitree-Go2-Velocity-Play` | Pure PPO playback and export |

```bash
# Train
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-AMP --num_envs 4096 --headless
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity --num_envs 4096 --headless

# Play and export
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-AMP-Play --num_envs 16
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-Velocity-Play --num_envs 16
```

Reduce `--num_envs` if GPU memory is limited. Use `--load_run` and
`--checkpoint` to select a model. Playback writes deployable models into the
selected run's `exported/` directory. Logs and models are ignored by Git.

## Expert motion data

Expert trajectories are intentionally **not distributed**. Only use authorized
data and place it locally under `datasets/motions/trot/*.json`. AMP training and
AMP playback require these files; pure PPO does not.

The frame schema is defined in
`source/himloco_lab/rsl_rl/datasets/motion_loader.py`: root pose, root
quaternion in `xyzw` order, joint positions, local foot positions, body
linear/angular velocities, and joint velocities. The AMP joint order is:

```text
FL hip/thigh/calf, FR hip/thigh/calf,
RL hip/thigh/calf, RR hip/thigh/calf
```

Validate retargeting and ordering before training:

```bash
python scripts/replay_data.py --task Unitree-Go2-AMP-Play --max_steps 500
```

## Current AMP and control settings

- Reward mix: `0.9 * task_reward + 0.1 * style_reward`.
- AMP style reward coefficient: `0.15`.
- Five stored history frames plus current: `6 x 45 = 270` actor inputs.
- Physics: 200 Hz (`dt = 0.005 s`); policy: 50 Hz (`decimation = 4`).
- Joint target: `default_joint_position + 0.25 * action`.
- Nominal gains: Kp 25 and Kd 0.5.

Algorithm settings live in
`source/himloco_lab/tasks/locomotion/agents/himloco_amp_rsl_rl_cfg.py`.
Environment, reward, command, terrain, and randomization settings live in
`source/himloco_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py`.

## MuJoCo sim-to-sim

```bash
python -m pip install mujoco pyyaml
python deploy/deploy_sim2sim/sim2sim.py
```

The newest exported Go2 AMP policy is selected automatically. Examples:

```bash
python deploy/deploy_sim2sim/sim2sim.py --policy /path/to/policy.pt
python deploy/deploy_sim2sim/sim2sim.py --headless --no-joystick --no-real-time --duration 2
```

See [the sim-to-sim guide](deploy/deploy_sim2sim/README.md) for observation and
joint ordering, controls, torque limits, and simulator differences. A working
sim-to-sim run does not establish hardware safety.

## Repository layout

```text
source/himloco_lab/             Isaac Lab extension and learning code
scripts/himloco_rsl_rl/         Training and playback entry points
scripts/replay_data.py          Expert-motion replay
deploy/deploy_sim2sim/          MuJoCo sim-to-sim
urdf/go2/                       Go2 description
datasets/                       Local-only expert data (ignored)
logs/ and outputs/              Local-only artifacts (ignored)
```

## Publication safety

This working tree descends from private research work. Never force-add expert
data, checkpoints, policies, logs, or secrets. Before publication, run:

```bash
python scripts/check_publication.py
```

The check covers the Git index and reachable commit trees. If inherited history
contains private data, publish from a separately reviewed clean-history
repository. Also review the staged diff and third-party asset licenses.

## License

Project code uses [LICENSE](LICENSE). Bundled Go2 assets retain the license in
`urdf/go2/LICENSE`.
