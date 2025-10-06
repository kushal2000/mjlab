"""Object configurations for manipulation tasks."""

import mujoco

from mjlab.entity import EntityCfg


def get_cube_spec(size: float = 0.02, mass: float = 0.1) -> mujoco.MjSpec:
  """Create a cube object specification.
  
  Args:
    size: Half-size of the cube (default: 0.02m = 2cm cube)
    mass: Mass of the cube in kg (default: 0.1kg)
    
  Returns:
    MjSpec with a free-floating cube.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body()
  body.name = "cube"
  freejoint = body.add_freejoint()
  freejoint.name = "cube_joint"
  
  # Add VISUAL geometry (no collision, for rendering only)
  visual_geom = body.add_geom()
  visual_geom.name = "cube_visual"
  visual_geom.type = mujoco.mjtGeom.mjGEOM_BOX
  visual_geom.size = [size, size, size]
  visual_geom.rgba = [0.8, 0.2, 0.2, 1.0]  # Bright red cube
  visual_geom.contype = 0  # No collision
  visual_geom.conaffinity = 0  # No collision
  visual_geom.group = 0  # Visual group (rendered)
  
  # Add COLLISION geometry (with physics properties)
  collision_geom = body.add_geom()
  collision_geom.name = "cube_collision"
  collision_geom.type = mujoco.mjtGeom.mjGEOM_BOX
  collision_geom.size = [size, size, size]
  collision_geom.mass = mass
  collision_geom.friction = [1.0, 0.005, 0.0001]  # [sliding, torsional, rolling]
  collision_geom.solref = [0.01, 1.0]  # Soft contact
  collision_geom.solimp = [0.998, 0.999, 0.001, 0.5, 2.0]  # Contact impedance (5 params)
  collision_geom.contype = 1  # Enable collision
  collision_geom.conaffinity = 1  # Enable collision
  collision_geom.group = 3  # Collision group (not rendered by default)
  
  return spec


def get_sphere_spec(radius: float = 0.02, mass: float = 0.1) -> mujoco.MjSpec:
  """Create a sphere object specification.
  
  Args:
    radius: Radius of the sphere (default: 0.02m)
    mass: Mass of the sphere in kg (default: 0.1kg)
    
  Returns:
    MjSpec with a free-floating sphere.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body()
  body.name = "sphere"
  freejoint = body.add_freejoint()
  freejoint.name = "sphere_joint"
  
  geom = body.add_geom()
  geom.name = "sphere_geom"
  geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
  geom.size = [radius]
  geom.mass = mass
  geom.rgba = [0.2, 0.8, 0.2, 1.0]  # Green sphere
  geom.friction = [1.0, 0.005, 0.0001]
  geom.solref = [0.01, 1.0]
  geom.solimp = [0.998, 0.999, 0.001, 0.5, 2.0]  # Contact impedance (5 params)
  
  return spec


def get_cylinder_spec(
  radius: float = 0.015, length: float = 0.04, mass: float = 0.1
) -> mujoco.MjSpec:
  """Create a cylinder object specification.
  
  Args:
    radius: Radius of the cylinder (default: 0.015m)
    length: Length of the cylinder (default: 0.04m)
    mass: Mass of the cylinder in kg (default: 0.1kg)
    
  Returns:
    MjSpec with a free-floating cylinder.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body()
  body.name = "cylinder"
  freejoint = body.add_freejoint()
  freejoint.name = "cylinder_joint"
  
  geom = body.add_geom()
  geom.name = "cylinder_geom"
  geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
  geom.size = [radius, length / 2.0]  # MuJoCo uses half-length
  geom.mass = mass
  geom.rgba = [0.2, 0.2, 0.8, 1.0]  # Blue cylinder
  geom.friction = [1.0, 0.005, 0.0001]
  geom.solref = [0.01, 1.0]
  geom.solimp = [0.998, 0.999, 0.001, 0.5, 2.0]  # Contact impedance (5 params)
  
  return spec

__all__ = [
  "get_cube_spec",
  "get_sphere_spec",
  "get_cylinder_spec",
]

