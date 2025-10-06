"""Reward functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from .commands import InHandReOrientationCommand


def success_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
  """Bonus reward for successfully reaching the goal.
  
  The object is considered to have reached the goal when the object orientation
  is within the threshold. The reward is 1.0 if the object has reached the goal,
  otherwise 0.0.
  
  Args:
    env: The environment object.
    command_name: The command term to be used for extracting the goal.
    object_cfg: The configuration for the scene entity. Default is "object".
    
  Returns:
    Binary reward tensor. Shape (num_envs,).
  """
  # Extract useful elements
  asset: Entity = env.scene[object_cfg.name]
  command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)
  
  # Obtain the goal orientation
  goal_quat_w = command_term.command[:, 3:7]
  # Obtain the threshold for the orientation error
  threshold = command_term.cfg.orientation_success_threshold
  # Calculate the orientation error
  dtheta = quat_error_magnitude(asset.data.root_link_quat_w, goal_quat_w)
  
  return (dtheta <= threshold).float()


def track_pos_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
  """Reward for tracking the object position using the L2 norm.
  
  The reward is the distance between the object position and the goal position.
  
  Args:
    env: The environment object.
    command_name: The command term to be used for extracting the goal.
    object_cfg: The configuration for the scene entity. Default is "object".
    
  Returns:
    L2 distance to goal. Shape (num_envs,).
  """
  # Extract useful elements
  asset: Entity = env.scene[object_cfg.name]
  command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)
  
  # Obtain the goal position
  goal_pos_e = command_term.command[:, 0:3]
  # Obtain the object position in the environment frame
  object_pos_e = asset.data.root_link_pos_w - env.scene.env_origins
  
  return torch.norm(goal_pos_e - object_pos_e, p=2, dim=-1)


def track_orientation_inv_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  rot_eps: float = 1e-3,
) -> torch.Tensor:
  """Reward for tracking the object orientation using the inverse of the orientation error.
  
  The reward is the inverse of the orientation error between the object orientation
  and the goal orientation.
  
  Args:
    env: The environment object.
    command_name: The command term to be used for extracting the goal.
    object_cfg: The configuration for the scene entity. Default is "object".
    rot_eps: The threshold for the orientation error. Default is 1e-3.
    
  Returns:
    Inverse orientation error. Shape (num_envs,).
  """
  # Extract useful elements
  asset: Entity = env.scene[object_cfg.name]
  command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)
  
  # Obtain the goal orientation
  goal_quat_w = command_term.command[:, 3:7]
  # Calculate the orientation error
  dtheta = quat_error_magnitude(asset.data.root_link_quat_w, goal_quat_w)
  
  return 1.0 / (dtheta + rot_eps)


def joint_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalty for joint velocities using L2 norm.
  
  Args:
    env: The environment object.
    asset_cfg: The configuration for the scene entity. Default is "robot".
    
  Returns:
    L2 norm of joint velocities. Shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  joint_vel = asset.data.joint_vel[:, jnt_ids]
  return torch.sum(torch.square(joint_vel), dim=-1)


def action_l2(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
  """Penalty for actions using L2 norm.
  
  Args:
    env: The environment object.
    action_name: The action term name. If None, uses all actions.
    
  Returns:
    L2 norm of actions. Shape (num_envs,).
  """
  if action_name is None:
    actions = env.action_manager.action
  else:
    actions = env.action_manager.get_term(action_name).raw_action
  return torch.sum(torch.square(actions), dim=-1)


def action_rate_l2(
  env: ManagerBasedRlEnv, action_name: str | None = None
) -> torch.Tensor:
  """Penalty for action rate using L2 norm.
  
  Args:
    env: The environment object.
    action_name: The action term name. If None, uses all actions.
    
  Returns:
    L2 norm of action rate. Shape (num_envs,).
  """
  if action_name is None:
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
  else:
    term = env.action_manager.get_term(action_name)
    actions = term.raw_action
    prev_actions = term.prev_raw_action
    
  action_rate = actions - prev_actions
  return torch.sum(torch.square(action_rate), dim=-1)


__all__ = [
  "success_bonus",
  "track_pos_l2",
  "track_orientation_inv_l2",
  "joint_vel_l2",
  "action_l2",
  "action_rate_l2",
]

