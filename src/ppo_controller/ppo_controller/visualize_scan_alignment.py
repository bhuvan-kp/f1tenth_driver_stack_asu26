"""
Offline sanity check -- run this on your laptop (no ROS needed) before
trusting the simulated lidar on the car.

It plots:
  1. The loaded occupancy grid (from map.yaml + its image)
  2. The centerline track (from the same CSV the observation node uses)
  3. A simulated scan from a chosen (x, y, yaw) pose, drawn as rays

If the centerline doesn't lie inside the track's free space, or the scan
rays don't terminate at the walls you'd expect, the map/track/OptiTrack
frames are not aligned (flipped axis, rotated origin, wrong resolution,
etc.) and must be fixed before deploying.

Usage:
    python3 visualize_scan_alignment.py \
        --map path/to/map.yaml \
        --centerline path/to/track_centerline.csv \
        --x 1.0 --y 2.0 --yaw 0.0
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from scan_simulator import MapDistanceField, simulate_scan
from track_utils import Track


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, help="path to map.yaml")
    parser.add_argument("--centerline", required=True, help="path to *_centerline.csv")
    parser.add_argument("--x", type=float, default=None, help="vehicle x (default: track start)")
    parser.add_argument("--y", type=float, default=None, help="vehicle y (default: track start)")
    parser.add_argument("--yaw", type=float, default=0.0, help="vehicle yaw, radians")
    parser.add_argument("--num-beams", type=int, default=64)
    parser.add_argument("--fov", type=float, default=4.7)
    parser.add_argument("--max-range", type=float, default=10.0)
    parser.add_argument("--eps", type=float, default=0.01)
    args = parser.parse_args()

    map_field = MapDistanceField(args.map)
    track = Track.from_centerline_file(args.centerline)

    x = args.x if args.x is not None else float(track.xs[0])
    y = args.y if args.y is not None else float(track.ys[0])
    yaw = args.yaw

    scan = simulate_scan(
        x, y, yaw, map_field,
        num_beams=args.num_beams, fov=args.fov,
        max_range=args.max_range, eps=args.eps,
    )

    fig, ax = plt.subplots(figsize=(9, 9))

    extent = [
        map_field.origin_x,
        map_field.origin_x + map_field.width * map_field.resolution,
        map_field.origin_y,
        map_field.origin_y + map_field.height * map_field.resolution,
    ]
    ax.imshow(
        ~map_field.occupied, cmap="gray", origin="lower", extent=extent, alpha=0.8
    )

    ax.plot(track.xs, track.ys, "b-", linewidth=1.5, label="centerline")

    beam_angles = yaw + np.linspace(-args.fov / 2.0, args.fov / 2.0, args.num_beams)
    for r, a in zip(scan, beam_angles):
        ax.plot([x, x + r * np.cos(a)], [y, y + r * np.sin(a)], "r-", linewidth=0.5)
    ax.plot(x, y, "go", markersize=10, label="vehicle pose")
    ax.arrow(x, y, 0.5 * np.cos(yaw), 0.5 * np.sin(yaw), head_width=0.15, color="g")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Map/track/scan alignment check\n"
        f"pose=({x:.2f}, {y:.2f}, {yaw:.2f} rad)  "
        f"scan range=[{scan.min():.2f}, {scan.max():.2f}] m"
    )
    ax.legend(loc="upper right")

    out_path = "scan_alignment_check.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(
        "Check that: (1) the blue centerline sits inside the white free "
        "space, not the black walls or outside the map entirely, and "
        "(2) red rays stop at wall boundaries rather than passing "
        "through them or stopping short in open space."
    )


if __name__ == "__main__":
    main()
