"""Run an exported Go2 or ChituMax policy in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from go2_model import load_go2_model


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "go2.yaml")
    parser.add_argument("--policy", type=Path, default=None, help="Override the TorchScript policy path.")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration in seconds.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--no-real-time", action="store_true", help="Do not pace headless simulation.")
    parser.add_argument(
        "--joystick", action=argparse.BooleanOptionalAction, default=True,
        help="Automatically use the first connected gamepad (default: enabled).",
    )
    parser.add_argument("--cmd", type=float, nargs=3, metavar=("VX", "VY", "WZ"), default=None)
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def newest_exported_policy(pattern: str) -> Path:
    policies = list(PROJECT_ROOT.glob(pattern))
    if not policies:
        raise FileNotFoundError(
            f"No policy matched {pattern}. Run play.py once, or pass --policy /path/to/policy.pt."
        )
    return max(policies, key=lambda path: path.stat().st_mtime)


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Return the world gravity direction expressed in the base frame."""
    qw, qx, qy, qz = quat_wxyz
    return np.asarray(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def named_joint_addresses(model: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos_adr, dof_adr = [], []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint: {name}")
        qpos_adr.append(model.jnt_qposadr[joint_id])
        dof_adr.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_adr), np.asarray(dof_adr)


def named_actuator_ids(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    result = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(matches) != 1:
            raise ValueError(f"Expected one actuator for {joint_name}, found {len(matches)}")
        result.append(matches[0])
    return np.asarray(result)


class JoystickCommand:
    """Read and scale a gamepad command while respecting training limits."""

    def __init__(self, enabled: bool, cfg: dict, ranges: dict):
        self.pygame = None
        self.joystick = None
        self.cfg = cfg
        self.ranges = ranges
        if not enabled:
            return
        try:
            import pygame

            # Initialize only the subsystems we use. pygame.init() also opens
            # audio and produces irrelevant ALSA errors on robot/headless PCs.
            pygame.display.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                print("[joystick] no gamepad detected; MuJoCo keyboard control remains available")
                pygame.quit()
                return
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            self.pygame = pygame
            self.joystick = joystick
            print(f"[joystick] connected: {joystick.get_name()}")
        except (ImportError, RuntimeError) as exc:
            print(f"[joystick] disabled: {exc}")

    @property
    def active(self) -> bool:
        return self.joystick is not None

    @staticmethod
    def _scale_axis(raw: float, deadzone: float, limits: list[float]) -> float:
        """Remove deadzone then map each half-axis to its asymmetric limit."""
        if abs(raw) <= deadzone:
            return 0.0
        value = np.sign(raw) * (abs(raw) - deadzone) / (1.0 - deadzone)
        return float(value * (limits[1] if value >= 0.0 else abs(limits[0])))

    def read(self) -> np.ndarray | None:
        if not self.active:
            return None
        self.pygame.event.pump()
        stop_button = int(self.cfg.get("stop_button", -1))
        if 0 <= stop_button < self.joystick.get_numbuttons() and self.joystick.get_button(stop_button):
            return np.zeros(3, dtype=np.float32)
        deadzone = float(self.cfg.get("deadzone", 0.08))
        result = np.zeros(3, dtype=np.float32)
        for index, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
            axis_cfg = self.cfg[name]
            axis = int(axis_cfg["axis"])
            if axis >= self.joystick.get_numaxes():
                raise ValueError(f"Gamepad has no axis {axis} required by joystick.{name}")
            raw = self.joystick.get_axis(axis) * float(axis_cfg.get("sign", 1.0))
            result[index] = self._scale_axis(raw, deadzone, self.ranges[name])
        return result

    def close(self) -> None:
        if self.pygame is not None:
            self.pygame.quit()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    policy_path = args.policy.expanduser().resolve() if args.policy else None
    if policy_path is None:
        if cfg.get("policy_path"):
            policy_path = resolve_path(cfg["policy_path"], ROOT)
        else:
            pattern = cfg.get("policy_glob", "logs/himloco_rsl_rl/go2_amp/*/exported/policy.pt")
            policy_path = newest_exported_policy(pattern)
    if cfg.get("xml_path"):
        xml_path = resolve_path(cfg["xml_path"], ROOT)
    elif cfg.get("urdf_path"):
        xml_path = resolve_path(cfg["urdf_path"], PROJECT_ROOT)
    else:
        raise ValueError("Config needs xml_path or urdf_path")
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy does not exist: {policy_path}")
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo model does not exist: {xml_path}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    policy = torch.jit.load(str(policy_path), map_location=device).eval()
    if cfg.get("xml_path"):
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    elif cfg.get("builder") == "chitu":
        from chitu_model import load_chitu_model

        model = load_chitu_model(xml_path, float(cfg["simulation_dt"]))
    else:
        model = load_go2_model(xml_path, float(cfg["simulation_dt"]))
    data = mujoco.MjData(model)

    model.opt.timestep = float(cfg["simulation_dt"])
    decimation = int(cfg["control_decimation"])
    duration = float(args.duration if args.duration is not None else cfg["simulation_duration"])
    joint_names = list(cfg["joint_names"])
    qpos_adr, dof_adr = named_joint_addresses(model, joint_names)
    actuator_ids = named_actuator_ids(model, joint_names)

    num_actions = int(cfg["num_actions"])
    num_obs = int(cfg["num_obs"])
    history_length = int(cfg["history_length"])
    if len(joint_names) != num_actions:
        raise ValueError("joint_names length must equal num_actions")

    policy_to_mujoco = np.asarray(cfg["policy_to_mujoco"], dtype=np.int64)
    mujoco_to_policy = np.argsort(policy_to_mujoco)
    default_q = np.asarray(cfg["default_angles"], dtype=np.float32)
    kp = np.asarray(cfg["kps"], dtype=np.float32)
    kd = np.asarray(cfg["kds"], dtype=np.float32)
    effort_limit = np.asarray(cfg["effort_limits"], dtype=np.float32)
    cmd_scale = np.asarray(cfg["cmd_scale"], dtype=np.float32)
    action_scale = np.asarray(cfg["action_scale"], dtype=np.float32)
    if action_scale.ndim == 0:
        action_scale = np.full(num_actions, float(action_scale), dtype=np.float32)
    if action_scale.shape != (num_actions,):
        raise ValueError(f"action_scale must be a scalar or {num_actions} values, got {action_scale.shape}")
    command = np.asarray(args.cmd if args.cmd is not None else cfg["cmd_init"], dtype=np.float32)
    ranges = cfg["command_ranges"]
    joystick = JoystickCommand(args.joystick, cfg.get("joystick", {}), ranges)

    # Set the same nominal state as Isaac Lab.
    data.qpos[2] = float(cfg["initial_height"])
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[qpos_adr] = default_q
    mujoco.mj_forward(model, data)

    action_policy = np.zeros(num_actions, dtype=np.float32)
    target_q = default_q.copy()
    obs_history = np.zeros(num_obs * history_length, dtype=np.float32)

    def set_command(new_command: np.ndarray) -> None:
        command[:] = new_command
        command[0] = np.clip(command[0], *ranges["lin_vel_x"])
        command[1] = np.clip(command[1], *ranges["lin_vel_y"])
        command[2] = np.clip(command[2], *ranges["ang_vel_z"])

    def key_callback(keycode: int) -> None:
        new_command = command.copy()
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "w": new_command[0] += 0.1
        elif key == "s": new_command[0] -= 0.1
        elif key == "a": new_command[1] += 0.1
        elif key == "d": new_command[1] -= 0.1
        elif key == "q": new_command[2] += 0.1
        elif key == "e": new_command[2] -= 0.1
        elif key == " ": new_command[:] = 0.0
        else: return
        set_command(new_command)

    def control_step(step: int) -> None:
        nonlocal action_policy, target_q, obs_history
        if step % decimation == 0:
            joystick_command = joystick.read()
            if joystick_command is not None:
                set_command(joystick_command)
            q = data.qpos[qpos_adr].astype(np.float32)
            dq = data.qvel[dof_adr].astype(np.float32)
            try:
                base_ang_vel = data.sensor("imu_gyro").data.copy().astype(np.float32)
                quat = data.sensor("imu_quat").data.copy().astype(np.float32)
            except KeyError:
                base_ang_vel = data.qvel[3:6].copy().astype(np.float32)
                quat = data.qpos[3:7].copy().astype(np.float32)

            one_obs = np.zeros(num_obs, dtype=np.float32)
            one_obs[0:3] = command * cmd_scale
            one_obs[3:6] = base_ang_vel * float(cfg["ang_vel_scale"])
            one_obs[6:9] = projected_gravity(quat)
            one_obs[9:21] = ((q - default_q) * float(cfg["dof_pos_scale"]))[policy_to_mujoco]
            one_obs[21:33] = (dq * float(cfg["dof_vel_scale"]))[policy_to_mujoco]
            one_obs[33:45] = action_policy
            np.clip(one_obs, -100.0, 100.0, out=one_obs)

            obs_history[num_obs:] = obs_history[:-num_obs]
            obs_history[:num_obs] = one_obs
            with torch.inference_mode():
                tensor = torch.from_numpy(obs_history).unsqueeze(0).to(device)
                action_policy = policy(tensor).cpu().numpy().reshape(-1).astype(np.float32)
            np.clip(action_policy, -100.0, 100.0, out=action_policy)
            action_mujoco = (action_policy * action_scale)[mujoco_to_policy]
            target_q = default_q + action_mujoco

            # Refresh at 10 Hz without flooding the terminal with new lines.
            policy_step = step // decimation
            policy_hz = 1.0 / (model.opt.timestep * decimation)
            display_interval = max(1, round(policy_hz / 10.0))
            if policy_step % display_interval == 0:
                print(
                    f"\r\033[2Kcmd: vx={command[0]:+5.2f} m/s  "
                    f"vy={command[1]:+5.2f} m/s  "
                    f"wz={command[2]:+5.2f} rad/s  "
                    f"height={data.qpos[2]:.3f} m",
                    end="",
                    flush=True,
                )

        q = data.qpos[qpos_adr]
        dq = data.qvel[dof_adr]
        raw_torque = (target_q - q) * kp - dq * kd
        curve = cfg.get("torque_speed_curve")
        if curve is None:
            torque = np.clip(raw_torque, -effort_limit, effort_limit)
        else:
            limit = np.where(raw_torque * dq > 0, curve["Y1"], curve["Y2"])
            falloff = np.clip((curve["X2"] - np.abs(dq)) / (curve["X2"] - curve["X1"]), 0, 1)
            limit = np.minimum(limit * falloff, effort_limit)
            torque = np.clip(raw_torque, -limit, limit)
        data.ctrl[actuator_ids] = torque
        mujoco.mj_step(model, data)

    print(f"policy: {policy_path}")
    print(f"model:  {xml_path}")
    print(f"physics: {1.0 / model.opt.timestep:.0f} Hz, policy: {1.0 / (model.opt.timestep * decimation):.0f} Hz")
    print("keys: W/S forward, A/D lateral, Q/E yaw, Space stop")
    print(
        "limits: "
        f"vx={ranges['lin_vel_x']}, vy={ranges['lin_vel_y']}, wz={ranges['ang_vel_z']}"
    )
    set_command(command)

    total_steps = int(duration / model.opt.timestep)
    try:
        if args.headless:
            for step in range(total_steps):
                start = time.perf_counter()
                control_step(step)
                if not args.no_real_time:
                    remaining = model.opt.timestep - (time.perf_counter() - start)
                    if remaining > 0.0:
                        time.sleep(remaining)
        else:
            from mujoco import viewer as mj_viewer

            with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
                step = 0
                while viewer.is_running() and step < total_steps:
                    start = time.perf_counter()
                    control_step(step)
                    viewer.sync()
                    remaining = model.opt.timestep - (time.perf_counter() - start)
                    if remaining > 0.0:
                        time.sleep(remaining)
                    step += 1
    finally:
        joystick.close()

    print(f"\r\033[2Kfinished: t={data.time:.3f}s, base_z={data.qpos[2]:.4f}m")


if __name__ == "__main__":
    main()
