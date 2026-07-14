"""
Self-contained (numpy/scipy/PIL only) simulated-lidar module.

Loads a standard ROS map_server-style map (YAML + image) and precomputes a
Euclidean distance transform (distance to nearest occupied cell, in meters,
for every free cell). At query time it "sphere traces" / marches each beam
outward, at every step jumping forward by exactly the local clearance
(read from the distance field with bilinear interpolation), until the
clearance drops below `eps` (a hit) or the accumulated range exceeds
`max_range`. This is the same style of distance-field ray marching that
`jax_pf.ray_marching.get_scan` uses during training (it also takes a
precomputed `distance_transform` array as input), so beam ranges here
should closely match what the policy saw in sim.

Assumes the map's yaw origin is zero (i.e. map x/y axes are axis-aligned
with the world/OptiTrack frame the CSV track and vehicle pose are given
in). If your map.yaml has a non-zero third `origin` component this module
will raise -- re-export/re-save the map with a zero yaw origin first.
"""

import pathlib
from typing import Tuple

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt


class MapDistanceField:
    """Loads a ROS map_server map (YAML + image) and exposes a bilinearly
    interpolated Euclidean distance-to-nearest-obstacle field, in meters."""

    def __init__(self, yaml_path):
        yaml_path = pathlib.Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"map yaml not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            map_meta = yaml.safe_load(f)

        image_path = (yaml_path.parent / map_meta["image"]).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"map image not found: {image_path}")

        self.resolution = float(map_meta["resolution"])
        origin = map_meta.get("origin", [0.0, 0.0, 0.0])
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        origin_theta = float(origin[2]) if len(origin) > 2 else 0.0
        if abs(origin_theta) > 1e-6:
            raise NotImplementedError(
                "Non-zero map yaw origin is not supported by this scan "
                "simulator. Re-save the map so origin[2] == 0, or extend "
                "world_to_grid()/sample_distance() to rotate into the map "
                "frame before this class is used."
            )

        negate = bool(map_meta.get("negate", 0))
        occupied_thresh = float(map_meta.get("occupied_thresh", 0.65))
        free_thresh = float(map_meta.get("free_thresh", 0.196))

        img = Image.open(image_path).convert("L")
        # row 0 of img_arr is the TOP of the image
        img_arr = np.asarray(img, dtype=np.float64) / 255.0

        # standard ROS map_server occupancy-probability convention
        occ_prob = img_arr if negate else (1.0 - img_arr)

        occupied = occ_prob > occupied_thresh
        free = occ_prob < free_thresh
        unknown = ~(occupied | free)
        # conservative: never let the simulated ray march through unmapped
        # territory
        occupied = occupied | unknown

        # image (0,0) is top-left; map (0,0) is bottom-left in world frame,
        # so flip vertically so that row index increases with world y
        occupied = np.flipud(occupied)

        self.height, self.width = occupied.shape
        self.occupied = occupied  # kept for diagnostics/plotting

        free_mask = (~occupied).astype(np.float64)
        # pixels distance to nearest occupied ("background") pixel
        dist_px = distance_transform_edt(free_mask)
        self.distance_transform = dist_px * self.resolution  # meters

    def world_to_grid(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx = (x - self.origin_x) / self.resolution
        gy = (y - self.origin_y) / self.resolution
        return gx, gy

    def sample_distance(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated clearance (meters) at fractional pixel
        coordinates gx (col), gy (row). Out-of-bounds points are clamped to
        the map edge (treated as whatever clearance the edge cell has --
        map padding/border should already be walled off if this matters)."""
        gx = np.clip(gx, 0.0, self.width - 1.0 - 1e-6)
        gy = np.clip(gy, 0.0, self.height - 1.0 - 1e-6)

        x0 = np.floor(gx).astype(int)
        y0 = np.floor(gy).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = gx - x0
        wy = gy - y0

        d00 = self.distance_transform[y0, x0]
        d10 = self.distance_transform[y0, x1]
        d01 = self.distance_transform[y1, x0]
        d11 = self.distance_transform[y1, x1]

        d0 = d00 * (1.0 - wx) + d10 * wx
        d1 = d01 * (1.0 - wx) + d11 * wx
        return d0 * (1.0 - wy) + d1 * wy

    def in_collision(self, x: float, y: float) -> bool:
        """Convenience check, e.g. for sanity-checking a pose before use."""
        gx, gy = self.world_to_grid(np.array([x]), np.array([y]))
        return bool(self.sample_distance(gx, gy)[0] <= 0.0)


def simulate_scan(
    x: float,
    y: float,
    theta: float,
    map_field: MapDistanceField,
    num_beams: int,
    fov: float,
    max_range: float,
    eps: float,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Vectorized (all beams at once) distance-field ray march.

    Beam angles are centered on `theta` (vehicle heading), spanning
    [-fov/2, +fov/2], matching the training-time convention. Returns an
    array of shape (num_beams,) of ranges in meters, clipped to
    [0, max_range].
    """
    beam_angles = theta + np.linspace(-fov / 2.0, fov / 2.0, num_beams)
    cos_t = np.cos(beam_angles)
    sin_t = np.sin(beam_angles)

    cur_x = np.full(num_beams, x, dtype=np.float64)
    cur_y = np.full(num_beams, y, dtype=np.float64)
    total_dist = np.zeros(num_beams, dtype=np.float64)
    active = np.ones(num_beams, dtype=bool)

    for _ in range(max_iterations):
        if not np.any(active):
            break

        gx, gy = map_field.world_to_grid(cur_x, cur_y)
        step = map_field.sample_distance(gx, gy)
        # never take a backwards/zero step for inactive beams
        step = np.where(active, np.maximum(step, 0.0), 0.0)

        total_dist = total_dist + step
        cur_x = cur_x + step * cos_t
        cur_y = cur_y + step * sin_t

        hit = step <= eps
        exceeded = total_dist >= max_range
        active = active & ~hit & ~exceeded

    return np.clip(total_dist, 0.0, max_range)
