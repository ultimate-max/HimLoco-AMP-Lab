# Go2 Sim2Sim

Run commands from the repository root in the training Python environment:

```bash
python -m pip install mujoco numpy torch pyyaml
# Optional gamepad support
python -m pip install pygame
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-AMP-Play --num_envs 1
python deploy/deploy_sim2sim/sim2sim.py
```

AMP play/export requires local motion data; MuJoCo execution does not.
The newest `logs/himloco_rsl_rl/go2_amp/*/exported/policy.pt` is selected
automatically. Alternatively specify `--policy /path/to/policy.pt`.
Compatible pure-RL Go2 exported actors can also be selected explicitly.

`config/go2.yaml` follows the current Go2 training configuration:

- Physics 200 Hz, policy 50 Hz, decimation 4.
- Kp 25, Kd 0.5, action scale 0.25 rad.
- Torque-speed limits X1=13.5, X2=30 rad/s, Y1=20.2, Y2=23.4 Nm.
- Policy joint positions, velocities and actions: FL/FR/RL/RR hips,
  then FL/FR/RL/RR thighs, then FL/FR/RL/RR calves.
- Internal control order: FL, FR, RL, RR, each hip/thigh/calf.
  YAML defines the permutation; MuJoCo joint addresses are resolved by name.
- Observations: command, body angular velocity, projected gravity,
  joint position offset, joint velocity, previous action: 45 values.
  Six latest-first frames give 270 inputs. Angular velocity scale is 0.25,
  joint velocity scale 0.05; other observation scales are 1.
- MuJoCo quaternion is wxyz; gravity projection uses this convention.
  The actor does not receive quaternion components directly.

Update the deployment configuration when changing training parameters.
The default model is the supplied `robot/go2/scene.xml`, including `go2.xml`
and its visual meshes. All asset paths are relative. This scene contains
obstacles; it is not a flat-only scene. Select another scene via `xml_path`.
The XML uses zero passive joint damping (PD runs in Python), zero added
armature, a 23.4 Nm motor limit, and the training nominal pose at 0.4 m.
Joint friction loss is 0.01 Nm as an approximation, not an exact conversion
of PhysX's dimensionless joint friction coefficient. Training inherits
armature from the imported asset; no additional armature is configured.
Sliding contact friction is fixed at 1 with three-dimensional contacts;
training instead randomizes static/dynamic coefficients in [0.2, 1.25]. The
Python controller follows the configured training PD and torque-speed limits,
but this does not make the two simulators' dynamics identical.
An optional URDF-conversion fallback remains available by removing `xml_path`
and setting `urdf_path: urdf/go2/urdf/go2_description.urdf` (repository-relative).
A smoke test does not establish walking performance or hardware safety.
Before public distribution, verify the supplied XML/mesh provenance and include
the applicable upstream license; permission to publish does not replace it.

Gamepad: left stick Y/X controls forward/lateral speed; right stick X controls
yaw. Button 0 zeros commands, not a hardware emergency stop. Configure axes,
signs and deadzone in YAML. Use `--no-joystick` to disable gamepad input.
Keyboard: W/S, A/D, Q/E, Space. Commands appear on one updating terminal line.
Limits match initial training ranges: vx/vy [-1, 1] m/s, yaw [-2, 2] rad/s.
Training forward-speed curriculum can expand further; deployment retains
the initial limits. `--cmd 0.3 0 0` sets an initial command.

```bash
python deploy/deploy_sim2sim/sim2sim.py --headless --no-joystick --no-real-time --duration 2
```

Checkpoints and exported policies are local-only; do not commit them.
