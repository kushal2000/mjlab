"""Observation functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_conjugate, quat_mul, quat_unique

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from .commands import InHandReOrientationCommand


def root_pos_w(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
  """Asset root position in the environment frame.
  
  Args:
    env: The environment object.
    asset_cfg: The configuration for the scene entity. Default is "object".
    
  Returns:
    Root position in environment frame. Shape (num_envs, 3).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w - env.scene.env_origins


def root_quat_w(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  make_quat_unique: bool = False,
) -> torch.Tensor:
  """Asset root orientation (w, x, y, z) in the environment frame.
  
  Args:
    env: The environment object.
    asset_cfg: The configuration for the scene entity. Default is "object".
    make_quat_unique: If True, ensures quaternion has non-negative real component.
    
  Returns:
    Root quaternion. Shape (num_envs, 4).
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.root_link_quat_w
  # Make the quaternion real-part positive if configured
  return quat_unique(quat) if make_quat_unique else quat


def root_lin_vel_w(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
  """Asset root linear velocity in the world frame.
  
  Args:
    env: The environment object.
    asset_cfg: The configuration for the scene entity. Default is "object".
    
  Returns:
    Root linear velocity. Shape (num_envs, 3).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w


def root_ang_vel_w(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
  """Asset root angular velocity in the world frame.
  
  Args:
    env: The environment object.
    asset_cfg: The configuration for the scene entity. Default is "object".
    
  Returns:
    Root angular velocity. Shape (num_envs, 3).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_w


def goal_quat_diff(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  make_quat_unique: bool,
) -> torch.Tensor:
  """Goal orientation relative to the asset's root frame.
  
  The quaternion is represented as (w, x, y, z). If make_quat_unique is True,
  the real part is always positive.
  
  Args:
    env: The environment object.
    asset_cfg: The configuration for the scene entity.
    command_name: The command term to be used for extracting the goal.
    make_quat_unique: If True, ensures quaternion has non-negative real component.
    
  Returns:
    Quaternion difference between goal and current orientation. Shape (num_envs, 4).
  """
  # Extract useful elements
  asset: Entity = env.scene[asset_cfg.name]
  command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)
  
  # Obtain the orientations
  goal_quat_w = command_term.command[:, 3:7]
  asset_quat_w = asset.data.root_link_quat_w
  
  # Compute quaternion difference
  quat = quat_mul(asset_quat_w, quat_conjugate(goal_quat_w))
  # Make sure the quaternion real-part is always positive
  return quat_unique(quat) if make_quat_unique else quat


def joint_pos_limit_normalized(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Joint positions normalized to [-1, 1] based on joint limits.
  
  Args:
    env: The environment object.
    asset_cfg: The configuration for the scene entity. Default is "robot".
    
  Returns:
    Normalized joint positions. Shape (num_envs, num_joints).
  """
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos[:, jnt_ids]
  joint_limits = asset.data.joint_pos_limits[:, jnt_ids]
  
  # Normalize to [-1, 1]
  joint_range = joint_limits[:, :, 1] - joint_limits[:, :, 0]
  joint_mid = (joint_limits[:, :, 1] + joint_limits[:, :, 0]) / 2.0
  normalized_pos = 2.0 * (joint_pos - joint_mid) / joint_range
  
  return normalized_pos


__all__ = [
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
  "goal_quat_diff",
  "joint_pos_limit_normalized",
]

