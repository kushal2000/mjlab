"""In-hand manipulation task configuration.

This module defines the base configuration for in-hand object reorientation tasks.
Robot-specific configurations should be defined in a config/ subdirectory.

This is adapted from Isaac Lab's in-hand manipulation task:
https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand
"""

from dataclasses import dataclass, field

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.viewer import ViewerConfig

##
# Scene definition
##

# TODO: Define robot configuration
# This should be an EntityCfg with:
# - spec_fn: Function that returns MjSpec for the robot (e.g., from XML file)
# - init_state: Initial joint positions and pose
# - articulation: Actuator configurations
# Example:
# ROBOT_CFG = EntityCfg(
#     spec_fn=lambda: mujoco.MjSpec.from_file("path/to/robot.xml"),
#     init_state=EntityCfg.InitialStateCfg(...),
#     articulation=EntityArticulationInfoCfg(...),
# )

# TODO: Define object configuration  
# This should be an EntityCfg for a manipulable object (e.g., cube)
# Example using a simple box:
# def get_cube_spec():
#     spec = mujoco.MjSpec()
#     body = spec.worldbody.add_body()
#     body.add_freejoint()
#     body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=0.1)
#     return spec
#
# OBJECT_CFG = EntityCfg(
#     spec_fn=get_cube_spec,
#     init_state=EntityCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.5),
#         rot=(1.0, 0.0, 0.0, 0.0),
#     ),
# )

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(terrain_type="plane"),
  num_envs=4096,
  env_spacing=0.6,
  entities={
    # "robot": ROBOT_CFG,  # TODO: Uncomment when ROBOT_CFG is defined
    # "object": OBJECT_CFG,  # TODO: Uncomment when OBJECT_CFG is defined
  },
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="",  # TODO: Set to appropriate body name (e.g., palm link)
  distance=1.5,
  elevation=-20.0,
  azimuth=45.0,
)

##
# MDP settings
##


@dataclass
class CommandsCfg:
  """Command specifications for the MDP."""
  object_pose: mdp.InHandReOrientationCommandCfg = term(
    mdp.InHandReOrientationCommandCfg,
    asset_name="object",
    resampling_time_range=(1.0e9, 1.0e9),  # Never resample based on time
    init_pos_offset=(0.0, 0.0, -0.04),
    update_goal_on_success=True,
    orientation_success_threshold=0.1,
    make_quat_unique=False,
    marker_pos_offset=(-0.2, -0.06, 0.08),
    debug_vis=True,
  )


@dataclass
class ActionsCfg:
  """Action specifications for the MDP."""
  joint_pos: mdp.JointPositionActionCfg = term(
    mdp.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=1.0,
    use_default_offset=True,
  )


@dataclass
class ObservationsCfg:
  """Observation specifications for the MDP."""

  @dataclass
  class KinematicObsGroupCfg(ObsGroup):
    """Observations with full-kinematic state information.
    
    This does not include acceleration or force information.
    """

    # Robot observations
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_pos_limit_normalized,
      noise=Gnoise(mean=0.0, std=0.005),
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_vel_rel,
      noise=Gnoise(mean=0.0, std=0.01),
    )

    # Object observations
    object_pos: ObsTerm = term(
      ObsTerm,
      func=mdp.root_pos_w,
      params={"asset_cfg": SceneEntityCfg("object")},
      noise=Gnoise(mean=0.0, std=0.002),
    )
    object_quat: ObsTerm = term(
      ObsTerm,
      func=mdp.root_quat_w,
      params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
    )
    object_lin_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.root_lin_vel_w,
      params={"asset_cfg": SceneEntityCfg("object")},
      noise=Gnoise(mean=0.0, std=0.002),
    )
    object_ang_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.root_ang_vel_w,
      params={"asset_cfg": SceneEntityCfg("object")},
      noise=Gnoise(mean=0.0, std=0.002),
    )

    # Command observations
    goal_pose: ObsTerm = term(
      ObsTerm,
      func=mdp.generated_commands,
      params={"command_name": "object_pose"},
    )
    goal_quat_diff: ObsTerm = term(
      ObsTerm,
      func=mdp.goal_quat_diff,
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "command_name": "object_pose",
        "make_quat_unique": False,
      },
    )

    # Action observations
    last_action: ObsTerm = term(ObsTerm, func=mdp.last_action)

    def __post_init__(self):
      self.enable_corruption = True
      self.concatenate_terms = True

  @dataclass
  class NoVelocityKinematicObsGroupCfg(KinematicObsGroupCfg):
    """Observations with partial kinematic state information.
    
    In contrast to the full-kinematic state group, this group does not include
    velocity information about the robot joints and the object root frame.
    """

    def __post_init__(self):
      # Call parent post init
      super().__post_init__()
      # Set unused terms to None
      self.joint_vel = None
      self.object_lin_vel = None
      self.object_ang_vel = None

  # Observation groups
  policy: KinematicObsGroupCfg = field(default_factory=KinematicObsGroupCfg)


@dataclass
class EventCfg:
  """Configuration for randomization."""
  reset_scene_to_default: EventTerm = term(
    EventTerm,
    func=mdp.reset_scene_to_default,
    mode="reset",
  )
  
  reset_robot_joints: EventTerm = term(
    EventTerm,
    func=mdp.reset_joints_within_limits_range,
    mode="reset",
    params={
      "position_range": {".*": (0.2, 0.2)},
      "velocity_range": {".*": (0.0, 0.0)},
      "use_default_offset": True,
      "operation": "scale",
    },
  )


@dataclass
class RewardsCfg:
  """Reward terms for the MDP."""

  # Task rewards
  track_orientation_inv_l2: RewTerm = term(
    RewTerm,
    func=mdp.track_orientation_inv_l2,
    weight=1.0,
    params={
      "object_cfg": SceneEntityCfg("object"),
      "rot_eps": 0.1,
      "command_name": "object_pose",
    },
  )
  success_bonus: RewTerm = term(
    RewTerm,
    func=mdp.success_bonus,
    weight=250.0,
    params={
      "object_cfg": SceneEntityCfg("object"),
      "command_name": "object_pose",
    },
  )

  # Penalties
  joint_vel_l2: RewTerm = term(
    RewTerm, func=mdp.joint_vel_l2, weight=-2.5e-5
  )
  action_l2: RewTerm = term(RewTerm, func=mdp.action_l2, weight=-0.0001)
  action_rate_l2: RewTerm = term(
    RewTerm, func=mdp.action_rate_l2, weight=-0.01
  )


@dataclass
class TerminationsCfg:
  """Termination terms for the MDP."""

  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)

  max_consecutive_success: DoneTerm = term(
    DoneTerm,
    func=mdp.max_consecutive_success,
    params={"num_success": 50, "command_name": "object_pose"},
  )

  object_out_of_reach: DoneTerm = term(
    DoneTerm,
    func=mdp.object_away_from_robot,
    params={"threshold": 0.3},
  )


##
# Simulation configuration
##

SIM_CFG = SimulationCfg(
  nconmax=500_000,  # Increased for hand-object contacts
  njmax=1000,  # Increased to accommodate hand-object contacts
  mujoco=MujocoCfg(
    timestep=1.0 / 120.0,  # 120 Hz simulation
    iterations=4,  # Fewer iterations than tracking task
    ls_iterations=10,
  ),
)

##
# Environment configuration
##


@dataclass
class InHandManipulationEnvCfg(ManagerBasedRlEnvCfg):
  """Configuration for the in-hand manipulation environment."""

  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationsCfg = field(default_factory=ObservationsCfg)
  actions: ActionsCfg = field(default_factory=ActionsCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)
  rewards: RewardsCfg = field(default_factory=RewardsCfg)
  terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
  events: EventCfg = field(default_factory=EventCfg)
  sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
  viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
  
  decimation: int = 4  # 30 Hz control frequency (120 Hz / 4)
  episode_length_s: float = 20.0


