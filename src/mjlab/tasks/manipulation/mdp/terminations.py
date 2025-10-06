"""Termination functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from .commands import InHandReOrientationCommand


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Check if the episode has timed out.
  
  Args:
    env: The environment object.
    
  Returns:
    Boolean tensor indicating timeout. Shape (num_envs,).
  """
  return env.episode_length_buf >= env.max_episode_length


def max_consecutive_success(
  env: ManagerBasedRlEnv, num_success: int, command_name: str
) -> torch.Tensor:
  """Check if the task has been completed consecutively for a certain number of times.
  
  Args:
    env: The environment object.
    num_success: Threshold for the number of consecutive successes required.
    command_name: The command term to be used for extracting the goal.
    
  Returns:
    Boolean tensor indicating max consecutive success. Shape (num_envs,).
  """
  command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)
  return command_term.metrics["consecutive_success"] >= num_success


def object_away_from_goal(
  env: ManagerBasedRlEnv,
  threshold: float,
  command_name: str,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
  """Check if object has gone far from the goal.
  
  The object is considered to be out-of-reach if the distance between the goal
  and the object is greater than the threshold.
  
  Args:
    env: The environment object.
    threshold: The threshold for the distance between the robot and the object.
    command_name: The command term to be used for extracting the goal.
    object_cfg: The configuration for the scene entity. Default is "object".
    
  Returns:
    Boolean tensor indicating if object is away from goal. Shape (num_envs,).
  """
  # Extract useful elements
  command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[object_cfg.name]
  
  # Object pos
  asset_pos_e = asset.data.root_link_pos_w - env.scene.env_origins
  goal_pos_e = command_term.command[:, :3]
  
  return torch.norm(asset_pos_e - goal_pos_e, p=2, dim=1) > threshold


def object_away_from_robot(
  env: ManagerBasedRlEnv,
  threshold: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
  """Check if object has gone far from the robot.
  
  The object is considered to be out-of-reach if the distance between the robot
  and the object is greater than the threshold.
  
  Args:
    env: The environment object.
    threshold: The threshold for the distance between the robot and the object.
    asset_cfg: The configuration for the robot entity. Default is "robot".
    object_cfg: The configuration for the object entity. Default is "object".
    
  Returns:
    Boolean tensor indicating if object is away from robot. Shape (num_envs,).
  """
  # Extract useful elements
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  
  # Compute distance
  dist = torch.norm(robot.data.root_link_pos_w - obj.data.root_link_pos_w, dim=1)
  
  return dist > threshold


__all__ = [
  "time_out",
  "max_consecutive_success",
  "object_away_from_goal",
  "object_away_from_robot",
]

