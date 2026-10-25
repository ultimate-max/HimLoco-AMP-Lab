"""ChituMax assets and canonical RobotLoad AMP data layout."""

from isaaclab.assets import ArticulationCfg

from .unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg
from .chitu_actuators import ChituActuatorCfg


from .chitu_data import ROBOTLOAD_ROOT, CHITU_JOINT_NAMES, CHITU_FOOT_NAMES, CHITU_FK_PARAMS, CHITU_MOTION_FILES

CHITU_MAX_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=str(ROBOTLOAD_ROOT / "resources/robots/chitu/urdf/chitu.urdf"),
        merge_fixed_joints=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ChituActuatorCfg(
            joint_names_expr=[".*_joint"],
            stiffness=300.0,
            damping=14.0,
            effort_limit={".*_HAA_joint": 280.0, ".*_HFE_joint": 280.0, ".*_KFE_joint": 299.0},
            velocity_limit=9.0,
            velocity_limit_sim=9.0,
            friction=0.0,
            friction_params_file=str(ROBOTLOAD_ROOT / "resources/friction/chitu_mgms_net_params.json"),
        ),
    },
    joint_sdk_names=CHITU_JOINT_NAMES,
)
