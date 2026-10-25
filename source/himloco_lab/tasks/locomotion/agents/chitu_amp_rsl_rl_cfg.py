"""RobotLoad AMP hyperparameters with the HimLoco actor/estimator interface."""

from isaaclab.utils import configclass
from himloco_lab.assets.chitu import CHITU_FK_PARAMS, CHITU_MOTION_FILES
from .himloco_amp_rsl_rl_cfg import AMPPPORunnerCfg


@configclass
class ChituAMPPPORunnerCfg(AMPPPORunnerCfg):
    experiment_name = "chitu_max_amp"
    history_length = 0
    max_iterations = 500000
    amp_motion_files = CHITU_MOTION_FILES
    amp_joint_reorder = "chitu"
    amp_foot_fk_params = CHITU_FK_PARAMS
    amp_recompute_joint_vel = True
    amp_num_preload_transitions = 2000000
    amp_reward_coef = 10.0
    amp_task_reward_lerp = 0.28
    min_normalized_std = {".*_HAA_joint": 0.05, ".*_HFE_joint": 0.02, ".*_KFE_joint": 0.05}

    def __post_init__(self):
        self.algorithm.symmetry_coef = 15.0
