import gymnasium as gym


for task_name, config_name in (("ChituMax-AMP", "RobotEnvCfg"), ("ChituMax-AMP-Play", "RobotPlayEnvCfg")):
    gym.register(
        id=task_name,
        entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.amp_env_cfg:{config_name}",
            "play_env_cfg_entry_point": f"{__name__}.amp_env_cfg:RobotPlayEnvCfg",
            "himloco_rsl_rl_cfg": "himloco_lab.tasks.locomotion.agents.chitu_amp_rsl_rl_cfg:ChituAMPPPORunnerCfg",
        },
    )
