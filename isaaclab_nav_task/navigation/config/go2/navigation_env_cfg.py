# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Go2 specific configuration for navigation environment."""

import os

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCameraCfg, patterns
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_nav_task.navigation.navigation_env_cfg import NavigationEnvCfg, ObservationsCfg
import isaaclab_nav_task.navigation.mdp as mdp

from isaaclab_nav_task.navigation.assets import GO2_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR  # isort: skip


LEG_JOINT_NAMES = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]


@configclass
class Go2ObservationsCfg(ObservationsCfg):
    """Go2-specific observation groups."""

    @configclass
    class LowLevelPolicyCfg(ObsGroup):
        """Observations for low-level policy, aligned with Go2 locomotion policy input."""
        velocity_commands = ObsTerm(
            # In navigation, low-level velocity targets come from processed high-level actions.
            func=mdp.generated_actions,
            clip=(-100, 100),
            params={"action_name": "velocity_command"},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(
            func=mdp.last_low_level_action,
            clip=(-100, 100),
            params={"action_term": "velocity_command"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    low_level_policy: LowLevelPolicyCfg = LowLevelPolicyCfg()


@configclass
class Go2NavigationEnvCfg(NavigationEnvCfg):
    observations: Go2ObservationsCfg = Go2ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator
        from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config

        initialize_depth_noise_generator(robot_name="go2", use_jit_precompiled=False)

        self.scene.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Go2-specific raycast camera setup
        self.scene.raycast_camera = RayCasterCameraCfg(
            mesh_prim_paths=["/World/ground"],
            update_period=0,
            prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.05), convention="world"  # 20 degrees rot=(0.9848078, 0.0, 0.1736482, 0.0)
            ),
            data_types=["distance_to_image_plane"],
            debug_vis=False,
            max_distance=11.0,
            pattern_cfg=patterns.PinholeCameraPatternCfg.from_ros_camera_info(
                # D435i camera parameters (640x480 native)
                fx=391.9765,
                fy=391.9765,
                cx=320.7953,
                cy=238.5750,
                width=640,
                height=480,
                downsample_factor=10,  # Downsample from 640x480 to 64x48
            ),
        )

        # Go2-only: switch to Him-compatible low-level action term.
        self.actions.velocity_command = mdp.PerceptiveNavigationSE2HimActionCfg(
            asset_name="robot",
            low_level_position_action=mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=LEG_JOINT_NAMES,
                scale=0.25,
                use_default_offset=True,
            ),
            low_level_decimation=4,
            observation_group="low_level_policy",
            policy_distr_type="gaussian",
            history_length=5,
            velocity_clip_min=[-1.0, -1.0, -1.0],
            velocity_clip_max=[1.0, 1.0, 1.0],
            low_level_encoder_onnx_file=os.path.join(
                ISAACLAB_NAV_TASKS_ASSETS_DIR,
                "Policies",
                "locomotion",
                "go2",
                "2026-04-25_16-09-15",
                "encoder.onnx",
            ),
            low_level_actor_onnx_file=os.path.join(
                ISAACLAB_NAV_TASKS_ASSETS_DIR,
                "Policies",
                "locomotion",
                "go2",
                "2026-04-25_16-09-15",
                "policy.onnx",
            ),
        )

        self.rewards.joint_acc_l2_joint.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
        }

        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh"]),
            "threshold": 1.0,
        }

        self.events.randomize_low_pass_filter_alpha = None
        self.events.randomize_action_scale = None

        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False

        # Go2-specific terrain mix: keep maze + flat + pits, remove stairs.
        terrain_gen = self.scene.terrain.terrain_generator
        if terrain_gen is not None:
            if "stairs" in terrain_gen.sub_terrains:
                terrain_gen.sub_terrains.pop("stairs")

            # Disable stair insertion inside maze tiles as well.
            if "maze" in terrain_gen.sub_terrains and hasattr(terrain_gen.sub_terrains["maze"], "add_stairs_to_maze"):
                terrain_gen.sub_terrains["maze"].add_stairs_to_maze = False

            # Re-normalize proportions for remaining terrain types.
            if "maze" in terrain_gen.sub_terrains:
                terrain_gen.sub_terrains["maze"].proportion = 0.4
            if "non_maze" in terrain_gen.sub_terrains:
                terrain_gen.sub_terrains["non_maze"].proportion = 0.3
            if "pits" in terrain_gen.sub_terrains:
                terrain_gen.sub_terrains["pits"].proportion = 0.3


@configclass
class Go2NavigationEnvCfg_DEV(Go2NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class Go2NavigationEnvCfg_PLAY(Go2NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class Go2NavigationEnvCfg_PLAY_FLAT(Go2NavigationEnvCfg_PLAY):
    """Play configuration with flat-only terrain (non_maze)."""

    def __post_init__(self):
        super().__post_init__()

        terrain_gen = self.scene.terrain.terrain_generator
        if terrain_gen is None:
            return

        # Keep only flat terrain for play.
        keys_to_remove = [name for name in terrain_gen.sub_terrains.keys() if name != "non_maze"]
        for key in keys_to_remove:
            terrain_gen.sub_terrains.pop(key)

        if "non_maze" in terrain_gen.sub_terrains:
            flat_cfg = terrain_gen.sub_terrains["non_maze"]
            flat_cfg.proportion = 1.0
            # Ensure flat terrain does not spawn random walls/obstacles.
            flat_cfg.randomize_wall = False
            flat_cfg.random_wall_ratio = 0.0

        # In maze_terrain(), obstacle density scales with difficulty,
        # so forcing zero difficulty yields pure flat ground.
        terrain_gen.difficulty_range = [0.0, 0.0]
