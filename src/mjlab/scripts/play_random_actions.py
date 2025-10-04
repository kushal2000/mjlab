"""Script to play with random actions."""

from pathlib import Path
from typing import Literal, cast

import gymnasium as gym
import numpy as np
import torch
import tyro
from typing_extensions import assert_never

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer


class RandomPolicy:
    """A simple random action policy."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def __call__(self, observations):
        """Generate random actions for the given observations."""
        batch_size = observations.shape[0]
        return torch.rand(batch_size, 12) * 2 - 1


def run_play(
  task: str,
  motion_file: str | None = None,
  num_envs: int | None = None,
  device: str | None = None,
  video: bool = False,
  video_length: int = 200,
  video_height: int | None = None,
  video_width: int | None = None,
  camera: int | str | None = None,
  render_all_envs: bool = True,
  viewer: Literal["native", "viser"] = "native",
):
  configure_torch_backends()

  if device is None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"[INFO]: Using device: {device}")

  env_cfg = cast(
    ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
  )

  if num_envs is not None:
    env_cfg.scene.num_envs = num_envs
  if camera is not None:
    env_cfg.sim.render.camera = camera
  if video_height is not None:
    env_cfg.sim.render.height = video_height
  if video_width is not None:
    env_cfg.sim.render.width = video_width

#   env = gym.make(
#     task, cfg=env_cfg, device=device, render_mode="rgb_array" if video else None
#   )
  env = gym.make(
    task, cfg=env_cfg, device=device
  )
  
  if video:
    print("[INFO] Recording videos during play")
    # Create a simple log directory for videos
    log_dir = Path("logs") / "random_play"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = gym.wrappers.RecordVideo(
      env,
      video_folder=str(log_dir / "videos" / "play"),
      step_trigger=lambda step: step == 0,
      video_length=video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env)

  # Create random policy
  policy = RandomPolicy(env.action_space)

  if viewer == "native":
    NativeMujocoViewer(env, policy, render_all_envs=render_all_envs).run()
  elif viewer == "viser":
    ViserViewer(env, policy, render_all_envs=render_all_envs).run()
  else:
    assert_never(viewer)
  env.close()


def main():
  """Entry point for the CLI."""
  tyro.cli(run_play)


if __name__ == "__main__":
  main()
