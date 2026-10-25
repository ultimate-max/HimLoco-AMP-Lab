"""Batched MGMS joint friction for Isaac Gym training.

Matches ``DeLaN_Stribeck/deploy/mgms_feedforward.hpp`` and
``RobotDynamics/FrictionModule/mgms.py`` (``mgms_friction``). Each physics
step integrates the presliding operator states ``δ_i`` and the driven
position low-pass. The model is

    H = B^{-1} Σ_i k_i δ_i
    F_C(τ_j) = f_c0 + f_c1 τ_j²
    F_v(ω) = direction · (Stribeck + √|ω| · thermal) + f_v1 ω + f_v2 |ω| ω + f_v3 ω³
    f = F_C(τ_j) · H + F_v(ω)
    τ_f = -f        (resistive torque, same sign convention as the C++ step())

Training runs at the reference temperature ``t0`` so ``thermal = 0``; the
``f_t1``/``f_t2`` terms are read but not exercised. ``direction_scale > 0``
uses ``tanh(ω / direction_scale)`` instead of ``sgn(ω)`` to avoid Stribeck
sign flipping near zero.

Only the scale on ``f_c1`` is randomized, once per episode:
``f_c1_eff = f_c1_nominal * scale``. The operator schedule, Stribeck and
viscous polynomial stay fixed. ``f_c1_nominal`` is the config value
``gms_f_c1`` when it is positive, otherwise the identified ``f_c1`` from the
JSON. Set ``gms_f_c1 = 0`` to use the identified value verbatim.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Tuple

import torch

_TAU_CLIP = 1.0e4
_TEMP_GUARD = 1e-8


def load_gms_json(path: str) -> dict:
    """Load an MGMS ``mgms_params.json``. Single joint is broadcast to all dofs."""
    with open(path, "r", encoding="utf-8") as handle:
        root = json.load(handle)
    joints = root.get("joints") or []
    if not joints:
        raise ValueError("{}: no joints".format(path))

    n_operators = int(root.get("n_operators", 4))
    r_delta = float(root.get("r_delta", 4.0))
    r_k = float(root.get("r_k", 2.0))
    direction_scale = float(root.get("direction_scale", 1e-2))
    position_mix = float(root.get("position_mix", 0.2))
    temperature_exponent = float(root.get("temperature_exponent", -3.0))

    def scalar(name: str, default: float = 0.0) -> torch.Tensor:
        rows = [float(joint.get(name, default)) for joint in joints]
        return torch.tensor(rows, dtype=torch.float32)

    params = {
        "n_operators": n_operators,
        "r_delta": r_delta,
        "r_k": r_k,
        "direction_scale": direction_scale,
        "position_mix": position_mix,
        "temperature_exponent": temperature_exponent,
        "f_c0": scalar("f_c0"),
        "f_c1": scalar("f_c1", 1e-4),
        "f_s": scalar("f_s"),
        "v_s": scalar("v_s", 0.05),
        "mu": scalar("mu", 1.0),
        "backlash": scalar("backlash", 1e-3),
        "f_t1": scalar("f_t1"),
        "f_t2": scalar("f_t2"),
        "t0": scalar("t0", 25.0),
        "f_v1": scalar("f_v1"),
        "f_v2": scalar("f_v2"),
        "f_v3": scalar("f_v3"),
        "n_param_joints": len(joints),
    }
    # Operator schedule Δ_i, k_i (shared across joints, identified by r_delta/r_k).
    index = torch.arange(1, n_operators + 1, dtype=torch.float32)
    deformation = (index / float(n_operators)).pow(r_delta)
    weight = (float(n_operators) + 1.0 - index).pow(r_k)
    stiffness = weight / (deformation * weight.sum())
    params["deformation_limit"] = deformation  # (M,)
    params["stiffness"] = stiffness            # (M,)
    return params


def _broadcast_per_dof(tensor: torch.Tensor, num_dof: int, n_src: int, device: torch.device) -> torch.Tensor:
    if n_src == 1:
        tensor = tensor.expand(num_dof)
    elif n_src != num_dof:
        raise ValueError("MGMS json has {} joints but the robot has {} dofs".format(n_src, num_dof))
    return tensor.to(device=device).contiguous()


class GmsFriction:
    """Per-env, per-dof MGMS. ``step`` returns τ_f with shape (num_envs, num_dof)."""

    def __init__(
        self,
        num_envs: int,
        num_dof: int,
        params: dict,
        device: torch.device,
        dt: float,
        f_c1: float = 0.0,
        f_c1_scale_range: Sequence[float] = (1.0, 1.0),
    ):
        if dt <= 0.0:
            raise ValueError("MGMS dt must be positive, got {}".format(dt))
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.n_operators = int(params["n_operators"])
        self.dt = float(dt)
        self.device = device
        self.position_mix = float(params["position_mix"])
        self.direction_scale = float(params["direction_scale"])
        self.temperature_exponent = float(params["temperature_exponent"])

        n_src = int(params["n_param_joints"])

        def per_dof(name: str) -> torch.Tensor:
            return _broadcast_per_dof(params[name], num_dof, n_src, device)

        self.f_c0 = per_dof("f_c0").view(1, num_dof)
        self.f_s = per_dof("f_s").view(1, num_dof)
        self.v_s = per_dof("v_s").clamp_min(_TEMP_GUARD).view(1, num_dof)
        self.mu = per_dof("mu").view(1, num_dof)
        self.backlash = per_dof("backlash").clamp_min(_TEMP_GUARD).view(1, num_dof)
        self.t0 = per_dof("t0").view(1, num_dof)
        self.f_t1 = per_dof("f_t1").view(1, num_dof)
        self.f_t2 = per_dof("f_t2").view(1, num_dof)
        self.f_v1 = per_dof("f_v1").view(1, num_dof)
        self.f_v2 = per_dof("f_v2").view(1, num_dof)
        self.f_v3 = per_dof("f_v3").view(1, num_dof)

        # Operator schedule (M,) -> broadcast over (num_dof, M)
        limit = params["deformation_limit"].to(device=device).contiguous()
        gain = params["stiffness"].to(device=device).contiguous()
        self.deformation_limit = limit.view(1, 1, self.n_operators)
        self.stiffness = gain.view(1, 1, self.n_operators)
        self.cap = (self.backlash.unsqueeze(-1) * self.deformation_limit)  # (1, D, M)

        # f_c1 nominal: config override > 0 wins, else identified JSON value.
        json_f_c1 = float(params["f_c1"].mean().item())
        self.f_c1_nominal = float(f_c1) if float(f_c1) > 0.0 else json_f_c1
        self.f_c1_scale_range = (float(f_c1_scale_range[0]), float(f_c1_scale_range[1]))

        n, d, m = num_envs, num_dof, self.n_operators
        self.delta = torch.zeros(n, d, m, device=device)
        self.driven = torch.zeros(n, d, device=device)
        self.has_driven = torch.zeros(n, d, dtype=torch.bool, device=device)
        self.f_c1 = torch.zeros(n, 1, device=device)
        self.resample_f_c1(torch.arange(num_envs, device=device))

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids is None or len(env_ids) == 0:
            return
        self.delta[env_ids] = 0.0
        self.driven[env_ids] = 0.0
        self.has_driven[env_ids] = False
        self.resample_f_c1(env_ids)

    def resample_f_c1(self, env_ids: torch.Tensor) -> None:
        """f_c1_eff = f_c1_nominal * scale. The scale is overwritten on every reset."""
        lo, hi = self.f_c1_scale_range
        if hi < lo:
            raise ValueError("f_c1 scale range must satisfy lo <= hi, got {}".format(self.f_c1_scale_range))
        if abs(hi - lo) < 1e-12:
            self.f_c1[env_ids] = self.f_c1_nominal * lo
            return
        scale = lo + (hi - lo) * torch.rand(len(env_ids), 1, device=self.device)
        self.f_c1[env_ids] = self.f_c1_nominal * scale

    def _viscous(self, qd: torch.Tensor) -> torch.Tensor:
        speed = qd.abs()
        stribeck = (self.f_s - self.f_c0) * torch.exp(
            -(speed / self.v_s).pow(self.mu)
        )
        # Training holds T = t0 so dtemp = 0 and thermal = 0. Skip the
        # signed_power(0, neg) branch; if f_t1/f_t2 are both 0 (current JSON)
        # thermal is 0 regardless.
        if float(self.f_t1.abs().sum().item()) == 0.0 and float(self.f_t2.abs().sum().item()) == 0.0:
            thermal = qd.new_zeros(qd.shape)
        else:
            dtemp = qd.new_zeros(qd.shape)  # temperature == t0 in training
            thermal = self.f_t1 * dtemp + self.f_t2 * self._signed_power(dtemp, self.temperature_exponent)
        if self.direction_scale > 0.0:
            direction = torch.tanh(qd / self.direction_scale)
        else:
            direction = torch.sign(qd)
        nonlinear = direction * (stribeck + speed.sqrt() * thermal)
        polynomial = self.f_v1 * qd + self.f_v2 * speed * qd + self.f_v3 * qd.pow(3)
        return nonlinear + polynomial

    @staticmethod
    def _signed_power(x: torch.Tensor, exponent: float) -> torch.Tensor:
        exp = float(exponent)
        magnitude = x.abs()
        if exp < 0.0:
            powered = magnitude.clamp(min=_TEMP_GUARD).pow(-exp).reciprocal()
            return torch.where(magnitude < _TEMP_GUARD, torch.zeros_like(powered), powered)
        odd = abs(exp - round(exp)) < 1e-8 and int(round(exp)) % 2 != 0
        powered = magnitude.pow(exp)
        if odd or exp < 0.0:
            return torch.sign(x) * powered
        return powered

    def step(self, q: torch.Tensor, qd: torch.Tensor, tau_j: Optional[torch.Tensor] = None) -> torch.Tensor:
        dt = self.dt
        mix = self.position_mix
        # Driven position low-pass and per-step displacement.
        virtual_prev = q - qd * dt
        prev = torch.where(self.has_driven, self.driven, virtual_prev)
        nxt = torch.where(self.has_driven, self.driven + mix * (q - self.driven),
                          virtual_prev + mix * (q - virtual_prev))
        self.driven = nxt
        self.has_driven = torch.ones_like(self.has_driven)
        displacement = nxt - prev  # (N, D)

        # δ_i update: candidate = displacement + δ; clip to ±B·Δ_i.
        candidate = displacement.unsqueeze(-1) + self.delta
        cap = self.cap  # (1, D, M) broadcasts over (N, D, M)
        self.delta = torch.sign(candidate) * torch.min(candidate.abs(), cap)

        # Hysteresis H = B^{-1} Σ k_i δ_i.
        hysteresis = (self.stiffness * self.delta).sum(dim=-1) / self.backlash  # (N, D)

        # Coulomb F_C(τ_j) = f_c0 + f_c1_eff τ_j².
        if tau_j is None:
            coulomb_scale = self.f_c0
        else:
            coulomb_scale = self.f_c0 + self.f_c1 * tau_j.square()
        coulomb = coulomb_scale * hysteresis

        viscous = self._viscous(qd)
        friction = coulomb + viscous
        tau_f = -friction
        return tau_f.clamp(-_TAU_CLIP, _TAU_CLIP)
