"""Allegro Hand robot configuration.

The Allegro Hand is a high-performance dexterous robotic hand with 16 DOF (4 fingers x 4 joints).
This configuration is adapted from MuJoCo Menagerie.

Reference:
- https://www.wonikrobotics.com/robot-hand
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie/tree/main/wonik_allegro
"""

from pathlib import Path

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# Paths
_CURRENT_DIR = Path(__file__).parent
_XML_DIR = _CURRENT_DIR / "xmls"
RIGHT_HAND_XML = _XML_DIR / "right_hand.xml"
LEFT_HAND_XML = _XML_DIR / "left_hand.xml"


def get_spec(hand_type: str = "right") -> mujoco.MjSpec:
  """Get Allegro hand MjSpec.
  
  Args:
    hand_type: Either "right" or "left" for the hand type.
    
  Returns:
    MjSpec for the Allegro hand.
  """
  if hand_type == "right":
    xml_path = RIGHT_HAND_XML
  elif hand_type == "left":
    xml_path = LEFT_HAND_XML
  else:
    raise ValueError(f"hand_type must be 'right' or 'left', got: {hand_type}")
  
  if not xml_path.exists():
    raise FileNotFoundError(
      f"Allegro hand XML not found at: {xml_path}\n"
      "Make sure the XML files are in the xmls/ directory."
    )
  
  return mujoco.MjSpec.from_file(str(xml_path))


# Initial state configuration
# Based on IsaacLab's Allegro hand configuration with thumb positioned for grasping
# NOTE: joint_pos is set to empty dict {} to avoid keyframe ctrl size issues when combining entities
# The default is {".*": 0.0} which would create a keyframe with ctrl, causing size mismatch in multi-entity scenes
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.5),
  # Rotation to orient hand properly (from IsaacLab)
  rot=(0.257551, 0.283045, 0.683330, -0.621782),
  joint_pos={},  # Empty dict to skip keyframe ctrl creation
  joint_vel={},  # Empty dict to skip keyframe vel
)

ALLEGRO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(),  # Empty tuple - XML already has actuators
  soft_joint_pos_limit_factor=1.0,
)

# Right hand configuration
ALLEGRO_RIGHT_HAND_CFG = EntityCfg(
  spec_fn=lambda: get_spec("right"),
  init_state=INIT_STATE,
  articulation=ALLEGRO_ARTICULATION,
)

# Left hand configuration
ALLEGRO_LEFT_HAND_CFG = EntityCfg(
  spec_fn=lambda: get_spec("left"),
  init_state=INIT_STATE,
  articulation=ALLEGRO_ARTICULATION,
)

# Action scale (for normalized actions)
# This can be tuned based on the task
ALLEGRO_ACTION_SCALE = 1.0


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(ALLEGRO_RIGHT_HAND_CFG)

  viewer.launch(robot.spec.compile())

