"""Explicit PD actuator with the RobotLoad MGMS plant-friction model."""

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass

from .gms_friction import GmsFriction, load_gms_json


class ChituActuator(IdealPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_model = GmsFriction(
            num_envs=self.computed_effort.shape[0],
            num_dof=self.computed_effort.shape[1],
            params=load_gms_json(cfg.friction_params_file),
            device=self.computed_effort.device,
            dt=cfg.physics_dt,
            f_c1_scale_range=(0.9, 1.1),
        )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.computed_effort.shape[0], device=self.computed_effort.device)[
                slice(None) if env_ids is None else env_ids
            ]
        self.friction_model.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        result = super().compute(control_action, joint_pos, joint_vel)
        self.applied_effort = self.applied_effort + self.friction_model.step(
            joint_pos, joint_vel, self.applied_effort
        )
        result.joint_efforts = self.applied_effort
        return result


@configclass
class ChituActuatorCfg(IdealPDActuatorCfg):
    class_type: type = ChituActuator
    friction_params_file: str = ""
    physics_dt: float = 0.005
