"""Allegro Hand in-hand manipulation environment configuration.

This configuration sets up an in-hand object reorientation task using the Allegro hand
from Wonik Robotics and a cube object.
"""

from dataclasses import dataclass, replace
from pathlib import Path

from mjlab.entity import EntityCfg
from mjlab.tasks.manipulation.inhand_env_cfg import InHandManipulationEnvCfg
from mjlab.tasks.manipulation.objects import get_cube_spec
from mjlab.asset_zoo.robots.allegro_hand.allegro_constants import ALLEGRO_RIGHT_HAND_CFG


@dataclass
class AllegroHandInHandEnvCfg(InHandManipulationEnvCfg):
  """Allegro Hand in-hand manipulation environment configuration.
  """

  def __post_init__(self):
    # Create Allegro hand configuration (uses built-in model from asset_zoo)
    allegro_cfg = ALLEGRO_RIGHT_HAND_CFG
    
    cube_cfg = EntityCfg(
      spec_fn=lambda: get_cube_spec(size=0.025, mass=0.08),  # 2.5cm half-size = 5cm cube
      init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.94),  # Very close to hand, just slightly above
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
      ),
    )
    # breakpoint()
    
    # Set up scene with robot and object
    self.scene.entities = {
      "robot": allegro_cfg,
      "object": cube_cfg,
    }
    
    # Simulation configuration - ensure limits are high enough
    self.sim.nconmax = 500_000
    self.sim.njmax = 1000
    
    # Viewer configuration - wider view to see the whole scene
    from mjlab.viewer import ViewerConfig
    self.viewer.origin_type = ViewerConfig.OriginType.WORLD
    self.viewer.distance = 1.2  # Pull back camera to see both hand and cube
    self.viewer.elevation = -25.0  # Look down from above
    self.viewer.azimuth = 45.0
    
    # Command configuration - keep cube at current position for reorientation
    self.commands.object_pose.init_pos_offset = (0.0, 0.0, 0.0)
    
    # Reward tuning for Allegro hand
    self.rewards.track_orientation_inv_l2.weight = 1.0
    self.rewards.success_bonus.weight = 250.0
    self.rewards.joint_vel_l2.weight = -2.5e-5
    self.rewards.action_l2.weight = -1e-4
    self.rewards.action_rate_l2.weight = -1e-2


@dataclass
class AllegroHandInHandEnvCfg_PLAY(AllegroHandInHandEnvCfg):
  """Allegro Hand in-hand manipulation environment for evaluation/play.
  
  This configuration disables randomization and extends episode length for evaluation.
  """

  def __post_init__(self):
    super().__post_init__()
    
    # Disable observation noise
    self.observations.policy.enable_corruption = False
    
    # Extend episode for longer manipulation attempts
    self.episode_length_s = 60.0
    
    # Reduce number of environments for visualization
    self.scene.num_envs = 1

