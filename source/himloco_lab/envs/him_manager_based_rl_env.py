import torch
from typing import Dict, Any
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse


class HimlocoManagerBasedRLEnv(ManagerBasedRLEnv):
    """HimLoco environment with Manager-Based RL interface."""
    
    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        """Initialize the HimLoco environment.
        
        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None.
            **kwargs: Additional keyword arguments (e.g., env_cfg_entry_point from gym.register).
        """
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._pre_pre_action = torch.zeros((self.num_envs, self.action_manager.total_action_dim), device=self.device)
        # Map the simulator's internal ordering to the canonical motion-file order.
        robot = self.scene.articulations["robot"]
        amp_joint_names = getattr(cfg, "amp_joint_names", None)
        amp_foot_names = getattr(cfg, "amp_foot_names", None)
        if amp_joint_names:
            self._amp_joint_indices, resolved_joint_names = robot.find_joints(amp_joint_names, preserve_order=True)
            if resolved_joint_names != amp_joint_names:
                raise ValueError(f"Could not resolve canonical AMP joints: {amp_joint_names}; got {resolved_joint_names}")
        else:
            self._amp_joint_indices = list(range(robot.data.joint_pos.shape[1]))
        if amp_foot_names:
            self._amp_foot_body_indices, resolved_foot_names = robot.find_bodies(amp_foot_names, preserve_order=True)
            if resolved_foot_names != amp_foot_names:
                raise ValueError(f"Could not resolve canonical AMP feet: {amp_foot_names}; got {resolved_foot_names}")
        else:
            self._amp_foot_body_indices, _ = robot.find_bodies(".*_foot")
        self.num_amp_obs = (len(self._amp_joint_indices) * 2 +  # joint_pos + joint_vel
                            len(self._amp_foot_body_indices) * 3 +  # foot_pos
                            3 + 3 + 1)  # base_lin_vel + base_ang_vel + z_pos

    def step(self, action: torch.Tensor) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Capture observations before reset.
        4. Perform rendering if gui is enabled.
        5. Update the environment counters and compute the rewards and terminations.
        6. Reset the environments that terminated.
        7. Compute the observations after reset.
        8. Return the observations (after reset), observations (before reset), rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing:
                - obs_buf: Current observations after reset (Dict[str, torch.Tensor])
                - obs_buf_before_reset: Observations before reset (Dict[str, torch.Tensor])
                - reward_buf: Rewards for each environment (torch.Tensor)
                - reset_terminated: Termination flags (torch.Tensor)
                - reset_time_outs: Timeout flags (torch.Tensor)
                - extras: Additional information dictionary (Dict[str, Any])
        """
        # process actions
        self._pre_pre_action = self.action_manager.prev_action.clone()
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # -- get observations before reset
        self.obs_buf_before_reset = self.observation_manager.compute()
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
            

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            # -- capture terminal AMP states before reset
            terminal_amp_states = self.get_amp_observations()[reset_env_ids]

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)
        else:
            terminal_amp_states = torch.zeros(0, self.num_amp_obs, device=self.device)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return (self.obs_buf, self.obs_buf_before_reset, self.reward_buf,
                self.reset_terminated, self.reset_time_outs, self.extras,
                reset_env_ids, self.obs_buf_before_reset.get("critic", self.obs_buf_before_reset["policy"]),
                terminal_amp_states)

    def _reset_idx(self, env_ids):
        """Reset environments and restore RSI commands after CommandManager reset."""
        super()._reset_idx(env_ids)
        pending = getattr(self, "_pending_amp_reset_command", None)
        if pending is None:
            return
        command_name, selected, lin_vel_xy, ang_vel_z, expert_heading = pending
        command = self.command_manager.get_term(command_name)
        command.vel_command_b[selected, :2] = lin_vel_xy
        command.vel_command_b[selected, 2] = ang_vel_z
        if command.cfg.heading_command:
            yaw_rate = torch.clamp(ang_vel_z, -0.5 * torch.pi, 0.5 * torch.pi)
            command.heading_target[selected] = torch.atan2(
                torch.sin(expert_heading + 2.0 * yaw_rate),
                torch.cos(expert_heading + 2.0 * yaw_rate),
            )
        self._pending_amp_reset_command = None
    @property
    def pre_pre_action(self) -> torch.Tensor:
        """The previous  previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._pre_pre_action
    
    def get_amp_observations(self) -> torch.Tensor:
        """Get AMP observations for discriminator training.
        
        Returns tensor of shape (num_envs, num_amp_obs):
            joint_pos + foot_pos + base_lin_vel + base_ang_vel + joint_vel + z_pos

        z_pos is root height above the mean terrain under the base, matching
        reference motions recorded on a ground plane at z=0.
        """
        robot = self.scene.articulations["robot"]
        joint_pos = robot.data.joint_pos[:, self._amp_joint_indices]
        joint_vel = robot.data.joint_vel[:, self._amp_joint_indices]
        base_lin_vel = robot.data.root_lin_vel_b
        base_ang_vel = robot.data.root_ang_vel_b
        # Use the small base scan, not the wide policy terrain scan, so nearby
        # stairs outside the base footprint do not bias the AMP height.
        terrain_hits = self.scene.sensors["base_height_scanner"].data.ray_hits_w[..., 2]
        valid_hits = torch.isfinite(terrain_hits)
        hit_count = valid_hits.sum(dim=1, keepdim=True)
        terrain_z = torch.where(valid_hits, terrain_hits, 0.0).sum(dim=1, keepdim=True)
        terrain_z = terrain_z / hit_count.clamp_min(1)
        # Rays can miss beyond terrain boundaries. Use the environment origin
        # elevation as a finite fallback when no local measurement is available.
        terrain_z = torch.where(hit_count > 0, terrain_z, self.scene.env_origins[:, 2:3])
        z_pos = robot.data.root_pos_w[:, 2:3] - terrain_z
        
        # Motion files store foot positions relative to the root, in root frame.
        foot_pos_w = robot.data.body_pos_w[:, self._amp_foot_body_indices, :]
        root_pos_w = robot.data.root_pos_w[:, None, :]
        root_quat_w = robot.data.root_quat_w[:, None, :].expand(-1, len(self._amp_foot_body_indices), -1)
        foot_pos = quat_apply_inverse(root_quat_w.reshape(-1, 4), (foot_pos_w - root_pos_w).reshape(-1, 3))
        foot_pos = foot_pos.reshape(self.num_envs, -1)
        
        return torch.cat([
            joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos
        ], dim=-1)
