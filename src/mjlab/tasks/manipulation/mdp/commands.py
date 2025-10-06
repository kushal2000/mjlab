"""Command generators for in-hand manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class InHandReOrientationCommand(CommandTerm):
  """Command generator for in-hand object reorientation.
  
  Generates random target orientations for the manipulated object.
  """

  cfg: InHandReOrientationCommandCfg

  def __init__(self, cfg: InHandReOrientationCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    
    # Get the object entity
    self.robot = env.scene.entities[cfg.asset_name]
    
    # Initialize command buffers (position + quaternion)
    self._command = torch.zeros(self.num_envs, 7, device=self.device)
    
    # Set initial position based on object's init state + offset
    init_pos = torch.tensor(self.robot.cfg.init_state.pos, device=self.device)
    init_pos_offset = torch.tensor(cfg.init_pos_offset, device=self.device)
    self._command[:, :3] = init_pos + init_pos_offset
    
    # Initialize with random orientations
    self._command[:, 3:] = self._sample_uniform_quaternions(self.num_envs)
    
    # Metrics
    self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Return the command: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]."""
    return self._command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample random target orientations for the given environments."""
    # Keep position the same, resample orientation
    self._command[env_ids, 3:] = self._sample_uniform_quaternions(len(env_ids))

  def _update_command(self) -> None:
    """Update command (no-op for this implementation)."""
    pass

  def _update_metrics(self) -> None:
    """Update metrics (placeholder for now)."""
    pass

  def _sample_uniform_quaternions(self, n: int) -> torch.Tensor:
    """Sample random quaternions uniformly from SO(3).
    
    Uses the algorithm from:
    "Uniform Random Rotations", Ken Shoemake, Graphics Gems III, 1992
    
    Args:
      n: Number of quaternions to sample
      
    Returns:
      Tensor of shape (n, 4) with quaternions in [w, x, y, z] format
    """
    # Sample three uniform random variables
    u1 = torch.rand(n, device=self.device)
    u2 = torch.rand(n, device=self.device) * 2 * torch.pi
    u3 = torch.rand(n, device=self.device) * 2 * torch.pi
    
    # Compute quaternion components
    sqrt_u1 = torch.sqrt(u1)
    sqrt_1_minus_u1 = torch.sqrt(1 - u1)
    
    w = sqrt_1_minus_u1 * torch.sin(u2)
    x = sqrt_1_minus_u1 * torch.cos(u2)
    y = sqrt_u1 * torch.sin(u3)
    z = sqrt_u1 * torch.cos(u3)
    
    return torch.stack([w, x, y, z], dim=-1)


@dataclass(kw_only=True)
class InHandReOrientationCommandCfg(CommandTermCfg):
  """Configuration for the uniform 3D orientation command term.
  
  This command generates random goal orientations for in-hand object reorientation.
  Goals are resampled based on success, not time.
  """
  
  # Required from CommandTermCfg
  class_type: type[CommandTerm] = InHandReOrientationCommand
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # Never resample based on time
  
  # Task-specific parameters
  asset_name: str = "object"
  """Name of the asset in the environment for which the commands are generated."""
  
  init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Position offset of the asset from its default position."""
  
  make_quat_unique: bool = False
  """Whether to make the quaternion unique by ensuring real part is positive."""
  
  orientation_success_threshold: float = 0.1
  """Threshold for the orientation error to consider the goal orientation to be reached."""
  
  update_goal_on_success: bool = True
  """Whether to update the goal orientation when the goal orientation is reached."""
  
  marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Position offset of the marker from the object's desired position."""
  debug_vis: bool = False
  """Whether to enable debug visualization."""


__all__ = ["InHandReOrientationCommand", "InHandReOrientationCommandCfg"]

