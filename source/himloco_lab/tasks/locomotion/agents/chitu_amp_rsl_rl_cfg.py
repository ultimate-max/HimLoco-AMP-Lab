"""RobotLoad AMP hyperparameters with the HimLoco actor/estimator interface."""

from isaaclab.utils import configclass
from himloco_lab.assets.chitu import CHITU_FK_PARAMS, CHITU_MOTION_FILES
from .himloco_amp_rsl_rl_cfg import AMPPPORunnerCfg


@configclass
class ChituAMPPPORunnerCfg(AMPPPORunnerCfg):
    experiment_name = "chitu_max_amp"
    # Current frame plus 5 past frames. The estimator reads all 6; the actor still
    # sees only the newest frame, estimated velocity, and latent.
    history_length = 5
    max_iterations = 500000
    amp_motion_files = CHITU_MOTION_FILES
    amp_joint_reorder = "chitu"
    amp_foot_fk_params = CHITU_FK_PARAMS
    amp_recompute_joint_vel = True
    amp_num_preload_transitions = 2000000
    amp_reward_coef = 12.0
    # Weight of the task reward in the mix. Style weight is 1 - this value.
    # 0.28 left style at about one third of the scalar reward because the task
    # term is several times larger per step; 0.14 brings the two closer.
    amp_task_reward_lerp = 0.8  # 1.0 - 0.8 - 0.5 - 0.28
    min_normalized_std = {".*_HAA_joint": 0.05, ".*_HFE_joint": 0.02, ".*_KFE_joint": 0.05}

    def __post_init__(self):
        # RobotLoad averages the KL over action dims and uses 0.5.
        self.algorithm.symmetry_coef = 1.0
