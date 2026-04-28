# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""HimLoco-compatible navigation SE2 action term with ONNX low-level inference."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.assets import check_file_path

if TYPE_CHECKING:
    from .navigation_se2_him_actions_cfg import PerceptiveNavigationSE2HimActionCfg


class PerceptiveNavigationSE2HimAction(ActionTerm):
    """Navigation action term compatible with HimLoco low-level ONNX policies.

    Inference path:
    obs_history -> encoder.onnx -> encoder_out -> policy.onnx([current_obs, encoder_out]).
    """

    cfg: PerceptiveNavigationSE2HimActionCfg

    _env: ManagerBasedRLEnv

    def __init__(self, cfg: PerceptiveNavigationSE2HimActionCfg, env: ManagerBasedRLEnv):
        ActionTerm.__init__(self, cfg, env)

        # Prepare low-level action terms.
        self.low_level_position_action_term: ActionTerm = self.cfg.low_level_position_action.class_type(
            cfg.low_level_position_action, env
        )

        # Navigation command action is always 3D: [vx, vy, omega].
        self._action_dim = 3
        self._init_buffers()

        self._prev_filtered_velocity_commands = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._low_pass_alpha = self.cfg.low_pass_filter_alpha
        self._enable_low_pass_filter = self.cfg.enable_low_pass_filter
        self._per_env_per_dim_low_pass_alpha = torch.full(
            (self.num_envs, self._action_dim), self._low_pass_alpha, device=self.device
        )

        # Navigation velocity clipping setup (optional)
        self._velocity_clip_min = None
        self._velocity_clip_max = None
        if self.cfg.velocity_clip_min is not None and self.cfg.velocity_clip_max is not None:
            self._velocity_clip_min = torch.tensor(
                self.cfg.velocity_clip_min, device=self.device, dtype=torch.float32
            )
            self._velocity_clip_max = torch.tensor(
                self.cfg.velocity_clip_max, device=self.device, dtype=torch.float32
            )

        if self.cfg.history_length < 0:
            raise ValueError(f"history_length must be >= 0, got {self.cfg.history_length}")

        if self.cfg.low_level_encoder_onnx_file is None or self.cfg.low_level_actor_onnx_file is None:
            raise ValueError(
                "PerceptiveNavigationSE2HimAction requires both low_level_encoder_onnx_file and "
                "low_level_actor_onnx_file."
            )
        if not check_file_path(self.cfg.low_level_encoder_onnx_file):
            raise FileNotFoundError(
                f"Encoder ONNX file '{self.cfg.low_level_encoder_onnx_file}' does not exist."
            )
        if not check_file_path(self.cfg.low_level_actor_onnx_file):
            raise FileNotFoundError(
                f"Actor ONNX file '{self.cfg.low_level_actor_onnx_file}' does not exist."
            )

        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise ImportError("PerceptiveNavigationSE2HimAction requires onnxruntime.") from exc

        providers = ["CPUExecutionProvider"]
        if str(self.device).startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._onnx_encoder_session = ort.InferenceSession(
            self.cfg.low_level_encoder_onnx_file,
            providers=providers,
        )
        self._onnx_actor_session = ort.InferenceSession(
            self.cfg.low_level_actor_onnx_file,
            providers=providers,
        )

        encoder_input = self._onnx_encoder_session.get_inputs()[0]
        self._onnx_encoder_input_name = encoder_input.name
        self._onnx_encoder_output_name = self._onnx_encoder_session.get_outputs()[0].name
        self._onnx_actor_input_name = self._onnx_actor_session.get_inputs()[0].name
        self._onnx_actor_output_name = self._onnx_actor_session.get_outputs()[0].name

        encoder_input_shape = encoder_input.shape
        if len(encoder_input_shape) < 2:
            raise ValueError(
                "Encoder ONNX input must have shape [batch, obs_dim]. "
                f"Got shape: {encoder_input_shape}"
            )
        try:
            encoder_obs_dim = int(encoder_input_shape[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Encoder ONNX input dim must be static [batch, obs_dim]. "
                f"Got shape: {encoder_input_shape}"
            ) from exc

        self._history_steps = self.cfg.history_length + 1
        if encoder_obs_dim % self._history_steps != 0:
            raise ValueError(
                "Encoder ONNX input dim is incompatible with history_length. "
                f"obs_dim={encoder_obs_dim}, history_steps={self._history_steps}"
            )

        self._one_step_obs_dim = encoder_obs_dim // self._history_steps
        self._history_obs_dim = encoder_obs_dim

        self._low_level_obs_history = torch.zeros(
            self.num_envs,
            self._history_obs_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._history_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _infer_actions_onnx(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Run HimLoco ONNX inference: history -> encoder, [current|encoder] -> actor."""
        obs_np = obs_history.detach().cpu().numpy().astype(np.float32)
        current_obs = obs_np[:, : self._one_step_obs_dim]
        num_envs = obs_np.shape[0]

        actions = []
        for i in range(num_envs):
            single_history = obs_np[i : i + 1]
            single_current = current_obs[i : i + 1]

            encoder_out = self._onnx_encoder_session.run(
                [self._onnx_encoder_output_name],
                {self._onnx_encoder_input_name: single_history},
            )[0]
            actor_input = np.concatenate([single_current, encoder_out], axis=1)
            action = self._onnx_actor_session.run(
                [self._onnx_actor_output_name],
                {self._onnx_actor_input_name: actor_input},
            )[0]
            actions.append(action)

        actions_np = np.concatenate(actions, axis=0)
        return torch.from_numpy(actions_np).to(self.device)

    def _stack_low_level_history(self, current_obs: torch.Tensor) -> torch.Tensor:
        """Stack low-level observations in flattened newest-first format expected by HimLoco."""
        if self.cfg.history_length == 0:
            return current_obs

        if current_obs.dim() == 1:
            current_obs = current_obs.unsqueeze(0)
        current_obs = current_obs[:, : self._one_step_obs_dim]

        # On first use (or after env reset), fill all history slots with current observation.
        uninitialized = ~self._history_initialized
        if torch.any(uninitialized):
            self._low_level_obs_history[uninitialized] = current_obs[uninitialized].repeat(1, self._history_steps)
            self._history_initialized[uninitialized] = True

        initialized = self._history_initialized
        if torch.any(initialized):
            self._low_level_obs_history[initialized] = torch.cat(
                (
                    current_obs[initialized],
                    self._low_level_obs_history[initialized, : -self._one_step_obs_dim],
                ),
                dim=-1,
            )

        return self._low_level_obs_history

    def apply_low_pass_filter(self, velocity_commands: torch.Tensor) -> torch.Tensor:
        """Apply low-pass filter to velocity commands for smoother locomotion."""
        if not self._enable_low_pass_filter:
            return velocity_commands

        alpha_values = self._per_env_per_dim_low_pass_alpha
        filtered_commands = (
            alpha_values * self._prev_filtered_velocity_commands
            + (1.0 - alpha_values) * velocity_commands
        )
        self._prev_filtered_velocity_commands.copy_(filtered_commands)
        return filtered_commands

    def process_actions(self, actions: torch.Tensor):
        """Process navigation actions for logging and low-level policy observations."""
        self._raw_navigation_velocity_actions[:] = actions
        if not self.cfg.use_raw_actions:
            self._processed_navigation_velocity_actions = (
                self._raw_navigation_velocity_actions * self._scale + self._offset
            )
        else:
            self._processed_navigation_velocity_actions[:] = self._raw_navigation_velocity_actions

        if self.cfg.policy_distr_type == "gaussian":
            self._processed_navigation_velocity_actions = torch.tanh(self._processed_navigation_velocity_actions)
        elif self.cfg.policy_distr_type == "beta":
            self._processed_navigation_velocity_actions = (self._processed_navigation_velocity_actions - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown policy distribution type: {self.cfg.policy_distr_type}")

        observations = self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
        base_lin_vel = observations[:, 0:3]
        vel_xyz = base_lin_vel.norm(dim=1, keepdim=True)

        self._processed_navigation_velocity_actions = (
            self._processed_navigation_velocity_actions + vel_xyz * self._policy_bias
        ) * self._policy_scaling
        self._processed_navigation_velocity_actions = self.apply_low_pass_filter(
            self._processed_navigation_velocity_actions
        )
        if self._velocity_clip_min is not None and self._velocity_clip_max is not None:
            self._processed_navigation_velocity_actions = torch.clamp(
                self._processed_navigation_velocity_actions,
                min=self._velocity_clip_min,
                max=self._velocity_clip_max,
            )

    @torch.inference_mode()
    def apply_actions(self):
        """Apply low-level actions with HimLoco-style ONNX encoder + actor inference."""
        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            self._prev_low_level_position_actions[:] = self._low_level_position_actions.clone()

            low_level_obs = self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
            low_level_policy_input = self._stack_low_level_history(low_level_obs)
            actions_phase = self._infer_actions_onnx(low_level_policy_input)

            position_action_dim = self.low_level_position_action_term.action_dim
            self._low_level_position_actions[:] = actions_phase[:, :position_action_dim]
            self.low_level_position_action_term.process_actions(self._low_level_position_actions)

        self.low_level_position_action_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids=env_ids)
        if self.cfg.history_length == 0:
            return

        if env_ids is None:
            env_ids = slice(None)
        self._low_level_obs_history[env_ids] = 0.0
        self._history_initialized[env_ids] = False

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_navigation_velocity_actions

    @property
    def filtered_velocity_commands(self) -> torch.Tensor:
        return self._prev_filtered_velocity_commands

    @property
    def low_pass_alpha_values(self) -> torch.Tensor:
        return self._per_env_per_dim_low_pass_alpha

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_position_actions

    @property
    def low_level_position_actions(self) -> torch.Tensor:
        return self._low_level_position_actions

    @property
    def prev_low_level_position_actions(self) -> torch.Tensor:
        return self._prev_low_level_position_actions

    def _init_buffers(self):
        self._raw_navigation_velocity_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros(
            (self.num_envs, self._action_dim), device=self.device
        )
        self._low_level_position_actions = torch.zeros(
            self.num_envs, self.low_level_position_action_term.action_dim, device=self.device
        )
        self._prev_low_level_position_actions = torch.zeros_like(self._low_level_position_actions)
        self._low_level_step_dt = self.cfg.low_level_decimation * self._env.physics_dt
        self._counter = 0
        self._scale = torch.tensor(self.cfg.scale, device=self.device, dtype=torch.float32)
        self._offset = torch.tensor(self.cfg.offset, device=self.device, dtype=torch.float32)
        self._policy_scaling = torch.tensor(self.cfg.policy_scaling, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._policy_bias = torch.zeros(self.num_envs, self._action_dim, device=self.device, dtype=torch.float32)
