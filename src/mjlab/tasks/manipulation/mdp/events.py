"""Event functions specific to the in-hand dexterous manipulation environments.

NOTE: These are simplified placeholder implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def reset_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
  for entity in env.scene.entities.values():
    if not isinstance(entity, Entity):
      continue
    print(f"Resetting entity to default")
    try:
      default_root_state = entity.data.default_root_state[env_ids].clone()
      default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
      entity.write_root_state_to_sim(default_root_state, env_ids=env_ids)
    except:
      print(f"Error writing root state to sim for entity")

    try:
      default_joint_pos = entity.data.default_joint_pos[env_ids].clone()
      default_joint_vel = entity.data.default_joint_vel[env_ids].clone()
      entity.write_joint_state_to_sim(
        default_joint_pos, default_joint_vel, env_ids=env_ids
      )
    except:
      print(f"Error writing joint state to sim for entity")

def reset_joints_within_limits_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  position_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  use_default_offset: bool = True,
  operation: str = "scale",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Reset joint positions and velocities within a range of the limits.
  
  Args:
    env: The environment object.
    env_ids: The environment IDs to reset.
    position_range: Range for joint positions as {regex: (min, max)}.
    velocity_range: Range for joint velocities as {regex: (min, max)}.
    use_default_offset: Whether to use default position as offset.
    operation: Either "scale" or "add" to apply the range.
    asset_cfg: The configuration for the scene entity.
  """
  # TODO: Implement proper joint resetting logic
  # This should:
  # 1. Match joint names using regex patterns
  # 2. Sample positions within specified ranges
  # 3. Apply operation (scale or add) relative to default or limits
  # 4. Set joint positions and velocities for specified env_ids
  asset: Entity = env.scene[asset_cfg.name]
  num_envs_to_reset = len(env_ids)
  
  # Placeholder: reset to default positions
  if asset.data.default_joint_pos is not None:
    default_pos = asset.data.default_joint_pos[env_ids]
    default_vel = asset.data.default_joint_vel[env_ids]
    asset.data.write_joint_state(default_pos, default_vel, env_ids=env_ids)


__all__ = ["reset_joints_within_limits_range", "reset_to_default"]

