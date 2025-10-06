"""Dexterous hand robot configurations for manipulation tasks."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg


def get_allegro_hand_spec(xml_path: str | Path | None = None) -> mujoco.MjSpec:
  """Load Allegro Hand specification.
  
  Args:
    xml_path: Path to Allegro hand XML file. If None, uses the built-in
              Allegro hand from asset_zoo.
    
  Returns:
    MjSpec for the Allegro hand.
  """
  if xml_path is None:
    # Use built-in Allegro hand from asset_zoo
    from mjlab.asset_zoo.robots.allegro_hand.allegro_constants import (
      ALLEGRO_RIGHT_HAND_CFG,
    )
    
    return ALLEGRO_RIGHT_HAND_CFG.spec_fn()
  
  xml_path = Path(xml_path)
  if not xml_path.exists():
    raise FileNotFoundError(f"Allegro hand XML not found at: {xml_path}")
  
  return mujoco.MjSpec.from_file(str(xml_path))


def create_allegro_hand_cfg(
  xml_path: str | Path | None = None, hand_type: str = "right"
) -> EntityCfg:
  """Create Allegro Hand entity configuration.
  
  Args:
    xml_path: Optional path to Allegro hand XML file. If None, uses built-in model.
    hand_type: Either "right" or "left" for the hand type (default: "right").
    
  Returns:
    EntityCfg for the Allegro hand.
  """
  if xml_path is None:
    # Use built-in configuration from asset_zoo
    from mjlab.asset_zoo.robots.allegro_hand.allegro_constants import (
      ALLEGRO_LEFT_HAND_CFG,
      ALLEGRO_RIGHT_HAND_CFG,
    )
    
    if hand_type == "right":
      return ALLEGRO_RIGHT_HAND_CFG
    elif hand_type == "left":
      return ALLEGRO_LEFT_HAND_CFG
    else:
      raise ValueError(f"hand_type must be 'right' or 'left', got: {hand_type}")
  else:
    # Load from custom XML path
    from mjlab.asset_zoo.robots.allegro_hand.allegro_constants import (
      ALLEGRO_ARTICULATION,
      INIT_STATE,
    )
    
    return EntityCfg(
      spec_fn=lambda: get_allegro_hand_spec(xml_path),
      init_state=INIT_STATE,
      articulation=ALLEGRO_ARTICULATION,
    )


__all__ = [
  "get_allegro_hand_spec",
  "create_allegro_hand_cfg",
]

