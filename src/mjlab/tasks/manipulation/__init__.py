"""In-hand manipulation tasks."""

from .hands import create_allegro_hand_cfg
from .inhand_env_cfg import InHandManipulationEnvCfg
from .objects import CUBE_CFG, CYLINDER_CFG, SPHERE_CFG

__all__ = [
  "InHandManipulationEnvCfg",
  "CUBE_CFG",
  "SPHERE_CFG",
  "CYLINDER_CFG",
  "create_allegro_hand_cfg",
]


