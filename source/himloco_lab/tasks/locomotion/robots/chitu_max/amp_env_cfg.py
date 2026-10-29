"""Isaac Lab port of AMP_for_hardware RobotLoad's ChituMax flat AMP task."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from himloco_lab.assets.chitu import (
    CHITU_MAX_CFG, CHITU_JOINT_NAMES, CHITU_FOOT_NAMES, CHITU_FK_PARAMS, CHITU_MOTION_FILES,
)
from himloco_lab.tasks.locomotion import mdp
from ..go2.velocity_env_cfg import RobotEnvCfg as BaseRobotEnvCfg
from .mdp import ChituVelocityCommandCfg, randomize_payload, robotload_reward


@configclass
class ChituRewardsCfg:
    robotload = RewTerm(func=robotload_reward, weight=1.0)


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    amp_joint_names = CHITU_JOINT_NAMES
    amp_foot_names = CHITU_FOOT_NAMES
    amp_foot_fk_params: dict = CHITU_FK_PARAMS
    rewards: ChituRewardsCfg = ChituRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = CHITU_MAX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 4096
        self.scene.env_spacing = 3.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.contact_forces.update_period = self.sim.dt
        self.actions.JointPositionAction.scale = {".*_HAA_joint": 0.18, ".*_HFE_joint": 0.3, ".*_KFE_joint": 0.3}
        self.commands.base_velocity = ChituVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=False,
            rel_standing_envs=0.1,
            debug_vis=False,
            ranges=ChituVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.0, 1.0),
            ),
        )
        self.curriculum.lin_vel_cmd_levels = None
        self.observations.policy.joint_vel_rel.noise.n_min = -0.8
        self.observations.policy.joint_vel_rel.noise.n_max = 0.8
        self.observations.critic.joint_vel_rel.noise.n_min = -0.8
        self.observations.critic.joint_vel_rel.noise.n_max = 0.8
        self.observations.critic.height_scanner = None
        self.events.physics_material.func = mdp.randomize_chitu_contact_materials
        self.events.physics_material.params.update(
            static_friction_range=(0.3, 1.0), dynamic_friction_range=(0.3, 1.0),
            restitution_range=(0.0, 0.5),
        )
        self.events.add_base_mass.params["mass_distribution_params"] = (-5.0, 5.0)
        self.events.randomize_rigid_body_com.params["com_range"] = {
            "x": (-0.1, 0.1), "y": (-0.08, 0.08), "z": (-0.08, 0.08),
        }
        self.events.link_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_hip", ".*_thigh", ".*_calf", ".*_foot"]),
                "mass_distribution_params": (0.9, 1.1), "operation": "scale",
            },
        )
        self.events.payload_mass = EventTerm(func=randomize_payload, mode="startup")
        self.events.payload_com = EventTerm(
            func=mdp.randomize_rigid_body_com, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="payload"),
                "com_range": {"x": (-0.1, 0.14), "y": (-0.1, 0.1), "z": (-0.15, 0.15)},
            },
        )
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1), "operation": "scale",
            },
        )
        # Global clock, same instant for every env. XY velocity is replaced, not added.
        self.events.push_robot.func = mdp.push_by_setting_xy_velocity
        self.events.push_robot.is_global_time = True
        self.events.push_robot.interval_range_s = (12.0, 12.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        # Constant base wrench for the whole episode, sampled once on reset.
        self.events.external_force = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "force_range": (-15.0, 15.0),
                "torque_range": (-10.0, 10.0),
            },
        )
        self.events.reference_state = EventTerm(
            func=mdp.reset_from_amp, mode="reset",
            params={
                # RobotLoad keeps reference_state_initialization_prob in the cfg but
                # reset_idx never reads it: every reset samples a reference frame.
                "motion_files": CHITU_MOTION_FILES, "probability": 1.0,
                "joint_names": CHITU_JOINT_NAMES, "joint_reorder": "chitu",
                "foot_fk_params": CHITU_FK_PARAMS, "recompute_joint_vel": True,
            },
        )
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["base", ".*_thigh", ".*_calf"],
        )


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        for event_name in (
            "physics_material", "add_base_mass", "randomize_rigid_body_com", "link_mass",
            "payload_mass", "payload_com", "actuator_gains", "push_robot", "external_force",
            "reference_state",
        ):
            setattr(self.events, event_name, None)
