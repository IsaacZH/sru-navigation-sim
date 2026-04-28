# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .navigation_se2_him_actions import PerceptiveNavigationSE2HimAction


@configclass
class PerceptiveNavigationSE2HimActionCfg(ActionTermCfg):
    """Configuration for HimLoco-compatible navigation SE2 action."""

    class_type: type[ActionTerm] = PerceptiveNavigationSE2HimAction

    low_level_decimation: int = 4
    """Decimation factor for applying the low-level action term."""

    use_raw_actions: bool = False
    """Whether to use raw actions or not."""

    scale: list[float] = [1.0, 1.0, 1.0]
    """Scale for the actions [vx, vy, w]."""

    offset: list[float] = [0.0, 0.0, 0.0]
    """Offset for the actions [vx, vy, w]."""

    low_level_position_action: ActionTermCfg = MISSING
    """Configuration of the low level position action term."""

    observation_group: str = "policy"
    """Observation group to use for the low level policy."""

    policy_scaling: list[float] = [1.0, 1.0, 1.0]
    """Policy dependent scaling for the actions [vx, vy, w]."""

    policy_distr_type: str = "gaussian"
    """Policy distribution type: 'gaussian', 'beta'."""

    enable_low_pass_filter: bool = True
    """Whether to enable low-pass filtering for velocity commands."""

    low_pass_filter_alpha: float = 0.5
    """Low-pass filter smoothing factor."""

    history_length: int = 0
    """Number of past low-level observation steps to stack with current step.

    history_length=0 means only current observation is used.
    history_length=5 means current + 5 past steps are used.
    """

    low_level_encoder_onnx_file: str | None = None
    """Path to low-level encoder ONNX file."""

    low_level_actor_onnx_file: str | None = None
    """Path to low-level actor/policy ONNX file."""

    velocity_clip_min: list[float] = None
    """Minimum clipping bounds for [vx, vy, omega]."""

    velocity_clip_max: list[float] = None
    """Maximum clipping bounds for [vx, vy, omega]."""
