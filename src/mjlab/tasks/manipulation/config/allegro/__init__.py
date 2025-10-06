"""Allegro hand configurations for in-hand manipulation."""

import gymnasium as gym

# Register training environment
gym.register(
  id="Mjlab-InHand-Allegro",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.inhand_env_cfg:AllegroHandInHandEnvCfg",
  },
)

# Register play/evaluation environment
gym.register(
  id="Mjlab-InHand-Allegro-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.inhand_env_cfg:AllegroHandInHandEnvCfg_PLAY",
  },
)

