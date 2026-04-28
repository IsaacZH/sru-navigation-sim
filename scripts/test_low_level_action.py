#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Smoke-test low-level locomotion via the navigation action term.

This script bypasses the navigation policy and directly feeds random
[vx, vy, omega] commands into the current "velocity_command" action term.
Commands are re-sampled every 4 seconds from ranges defined in this file.

Example:
    python scripts/test_low_level_action.py --task Isaac-Nav-PPO-Go2-Play-v0 --num_envs 1 --steps 3000
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


# Command sampling configuration.
CMD_RESAMPLE_INTERVAL_S = 4.0
VX_RANGE = (-0.8, 0.8)
VY_RANGE = (-0.5, 0.5)
OMEGA_RANGE = (-1.0, 1.0)


parser = argparse.ArgumentParser(description="Test low-level locomotion with constant velocity targets.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during the test.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--task", type=str, required=True, help="Task name (e.g. Isaac-Nav-PPO-Go2-Play-v0).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=3000, help="Number of simulation steps to run.")
parser.add_argument("--report_interval", type=int, default=100, help="How often to print tracking stats (steps).")
parser.add_argument(
    "--follow_robot_view",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Default viewer tracks robot root (asset_root).",
)
parser.add_argument(
    "--flat_terrain",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use near-flat generated terrain (keeps height-field metadata required by goal sampling).",
)
parser.add_argument(
    "--disable_pushes",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable external push/disturbance events during test (default: enabled).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Enable cameras only when recording video.
args_cli.enable_cameras = bool(args_cli.video)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg


def _configure_env(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Apply test-oriented config overrides."""
    env_cfg.scene.num_envs = args_cli.num_envs

    # Default camera behavior: follow robot root in environment 0.
    if args_cli.follow_robot_view and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0

    # Keep generated terrain metadata because navigation goal sampling depends on it.
    # Match Play-Flat behavior: keep only non_maze terrain and force zero obstacle difficulty.
    if args_cli.flat_terrain and env_cfg.scene.terrain is not None:
        env_cfg.scene.terrain.max_init_terrain_level = 0
        if env_cfg.scene.terrain.terrain_generator is not None:
            terrain_gen = env_cfg.scene.terrain.terrain_generator
            terrain_gen.curriculum = False
            terrain_gen.num_rows = 1
            terrain_gen.num_cols = 1

            # Keep only flat sub-terrain.
            if hasattr(terrain_gen, "sub_terrains") and terrain_gen.sub_terrains is not None:
                keys_to_remove = [name for name in terrain_gen.sub_terrains.keys() if name != "non_maze"]
                for key in keys_to_remove:
                    terrain_gen.sub_terrains.pop(key)

                if "non_maze" in terrain_gen.sub_terrains:
                    flat_cfg = terrain_gen.sub_terrains["non_maze"]
                    flat_cfg.proportion = 1.0
                    if hasattr(flat_cfg, "randomize_wall"):
                        flat_cfg.randomize_wall = False
                    if hasattr(flat_cfg, "random_wall_ratio"):
                        flat_cfg.random_wall_ratio = 0.0

            if hasattr(terrain_gen, "difficulty_range"):
                terrain_gen.difficulty_range = [0.0, 0.0]

    if args_cli.disable_pushes and hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None

    # Reduce observation noise to make command-tracking easier to interpret.
    if hasattr(env_cfg, "observations"):
        if hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg.observations, "low_level_policy"):
            env_cfg.observations.low_level_policy.enable_corruption = False


def main() -> None:
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")

    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    _configure_env(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    obs, _ = env.reset()

    action_term = env.unwrapped.action_manager.get_term("velocity_command")
    action_dim = int(action_term.action_dim)
    if action_dim != 3:
        raise RuntimeError(f"Expected velocity_command action dim=3, got {action_dim}.")

    device = env.unwrapped.device
    total_steps = args_cli.steps
    step_dt = getattr(env.unwrapped, "step_dt", env_cfg.decimation * env_cfg.sim.dt)
    resample_steps = max(1, int(round(CMD_RESAMPLE_INTERVAL_S / step_dt)))

    def sample_commands() -> torch.Tensor:
        sampled = torch.empty((env.unwrapped.num_envs, 3), dtype=torch.float32, device=device)
        sampled[:, 0].uniform_(VX_RANGE[0], VX_RANGE[1])
        sampled[:, 1].uniform_(VY_RANGE[0], VY_RANGE[1])
        sampled[:, 2].uniform_(OMEGA_RANGE[0], OMEGA_RANGE[1])
        return sampled

    actions = sample_commands()

    print("[INFO] Low-level action test started")
    print(f"[INFO] task={args_cli.task} | num_envs={env.unwrapped.num_envs} | steps={total_steps}")
    print(f"[INFO] video={args_cli.video} | video_length={args_cli.video_length}")
    print(f"[INFO] follow_robot_view={args_cli.follow_robot_view}")
    print(f"[INFO] command resample interval: {CMD_RESAMPLE_INTERVAL_S:.2f}s ({resample_steps} env steps)")
    print(f"[INFO] ranges: vx={VX_RANGE}, vy={VY_RANGE}, omega={OMEGA_RANGE}")

    for step in range(1, total_steps + 1):
        if step == 1 or (step - 1) % resample_steps == 0:
            actions = sample_commands()
            mean_cmd = actions.mean(dim=0)
            print(
                f"[INFO] step={step}: resampled cmd mean="
                f"({mean_cmd[0].item():+.3f}, {mean_cmd[1].item():+.3f}, {mean_cmd[2].item():+.3f})"
            )

        with torch.inference_mode():
            obs, _, terminated, truncated, _ = env.step(actions)

        if not simulation_app.is_running():
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
