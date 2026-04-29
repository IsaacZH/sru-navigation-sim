#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Export runtime terrain data from an Isaac Lab navigation task.

This script is intentionally headless by default. It creates the task environment,
triggers terrain generation, and exports the stored height-field tensors injected
by the terrain patches in isaaclab_nav_task.

Usage:
    python scripts/export_terrain.py --task Isaac-Nav-PPO-Go2-Play-v0 --output terrain_dump.npz

Optional single-tile export:
    python scripts/export_terrain.py \
        --task Isaac-Nav-PPO-Go2-Play-v0 \
        --output terrain_dump.npz \
        --tile-out tile0.npy \
        --tile-index 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export terrain tensors from an Isaac Lab navigation task.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--output", type=str, required=True, help="Output .npz file path.")
parser.add_argument("--tile-out", type=str, default=None, help="Optional output .npy path for one terrain tile.")
parser.add_argument("--tile-index", type=int, default=0, help="Terrain tile index for --tile-out.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _hydra_args = parser.parse_known_args()

# Terrain export does not need cameras or a rendered viewport.
args_cli.enable_cameras = False
if not getattr(args_cli, "headless", False):
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching app.
import gymnasium as gym  # type: ignore[import-not-found]
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch tensors and array-likes to numpy arrays."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _extract_terrain_meta(task_name: str, env_cfg: ManagerBasedRLEnvCfg, terrain, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Build metadata that helps downstream conversion to MuJoCo hfield."""
    meta: dict[str, Any] = {
        "task": task_name,
    }

    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
    if generator_cfg is not None:
        for key in [
            "size",
            "num_rows",
            "num_cols",
            "horizontal_scale",
            "vertical_scale",
            "border_width",
        ]:
            if hasattr(generator_cfg, key):
                value = getattr(generator_cfg, key)
                if isinstance(value, tuple):
                    value = list(value)
                meta[key] = value

    origins = getattr(terrain, "terrain_origins", None)
    origins_np = _to_numpy(origins)
    if origins_np is not None:
        arrays["terrain_origins"] = origins_np
        meta["terrain_origins_shape"] = list(origins_np.shape)

    for name, array in arrays.items():
        if array is not None:
            meta[f"{name}_shape"] = list(array.shape)
            meta[f"{name}_dtype"] = str(array.dtype)

    return meta


def main() -> None:
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    if env_cfg_class is None:
        raise RuntimeError(f"Task '{args_cli.task}' does not expose env_cfg_entry_point.")

    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    try:
        env.reset()

        terrain = getattr(getattr(env.unwrapped, "scene", None), "terrain", None)
        if terrain is None:
            raise RuntimeError("Environment scene terrain is unavailable.")

        arrays: dict[str, np.ndarray] = {
            "height_field_visual": _to_numpy(getattr(terrain, "_height_field_visual", None)),
            "height_field_valid_mask": _to_numpy(getattr(terrain, "_height_field_valid_mask", None)),
            "height_field_platform_mask": _to_numpy(getattr(terrain, "_height_field_platform_mask", None)),
            "height_field_spawn_mask": _to_numpy(getattr(terrain, "_height_field_spawn_mask", None)),
        }

        if arrays["height_field_visual"] is None:
            raise RuntimeError(
                "No _height_field_visual found on terrain. Ensure isaaclab_nav_task patches are active for this task."
            )

        meta = _extract_terrain_meta(args_cli.task, env_cfg, terrain, arrays)
        arrays["meta_json"] = np.array(json.dumps(meta), dtype=np.unicode_)

        output_path = Path(args_cli.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)
        print(f"[INFO] Saved terrain dump to: {output_path}")

        if args_cli.tile_out:
            visual = arrays["height_field_visual"]
            if visual.ndim < 3:
                raise RuntimeError(f"Expected height_field_visual with at least 3 dims, got shape {visual.shape}")

            tile_count = int(visual.shape[0])
            tile_index = int(args_cli.tile_index)
            if tile_index < 0 or tile_index >= tile_count:
                raise IndexError(f"tile-index {tile_index} is out of range [0, {tile_count - 1}]")

            tile_out_path = Path(args_cli.tile_out)
            tile_out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(tile_out_path, visual[tile_index])
            print(f"[INFO] Saved tile {tile_index} to: {tile_out_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
