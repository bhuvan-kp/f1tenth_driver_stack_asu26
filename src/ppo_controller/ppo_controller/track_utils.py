"""
Self-contained (numpy/scipy only) port of the training-time Track /
CubicSplineND classes.

This intentionally drops everything the ROS observation node doesn't need:
  - jax / jax.numpy (no jit, no device dependency)
  - the `requests` + tarfile map auto-download path
  - occupancy-grid / PIL image loading
  - raceline-file-only constructors, from_numpy, etc.

What's kept is exactly what the node needs to reproduce the training-time
Frenet conversion bit-for-bit (up to solver/library differences):
  Track.from_track_dir(...), cartesian_to_frenet, frenet_to_cartesian,
  curvature.

Track data (the `<stem>_centerline.csv` / `<stem>_raceline.csv` / `<stem>.yaml`
files) is expected to already live on disk -- e.g. vendored inside your ROS
package's share directory -- so nothing is downloaded at runtime.
"""

import pathlib
from typing import Optional

import numpy as np
from scipy import interpolate


def _validate_waypoint_table(
    waypoints: np.ndarray,
    description: str,
    min_columns: int,
    column_description: str,
) -> np.ndarray:
    waypoints = np.asarray(waypoints)
    if waypoints.ndim != 2:
        raise ValueError(f"{description} must be a 2-dimensional array.")
    if waypoints.shape[0] < 2:
        raise ValueError(f"{description} must contain at least two rows.")
    if waypoints.shape[1] < min_columns:
        raise ValueError(f"expected {description} columns as {column_description}")
    return waypoints


def _validate_reference_points(xs: np.ndarray, ys: np.ndarray) -> None:
    if xs.shape[0] < 2:
        raise ValueError("track must contain at least two points.")
    if not np.any(np.hypot(np.diff(xs), np.diff(ys)) > 0):
        raise ValueError("track must contain at least two distinct points.")


def _calc_yaw_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    return np.arctan2(dy_dt, dx_dt)


def nearest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray) -> tuple:
    """Numpy port of nearest_point_on_trajectory_jax -- same math, no jax."""
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.sum((point - trajectory[:-1, :]) * diffs[:, :], axis=1)
    t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t[:, None] * diffs)
    dists = np.linalg.norm(point - projections, axis=1)
    min_dist_segment = int(np.argmin(dists))
    return dists[min_dist_segment], t[min_dist_segment], min_dist_segment


class CubicSplineND:
    """Numpy/scipy-only cubic spline. Mirrors the jax CubicSplineND closely
    enough that calc_position / calc_yaw / calc_curvature / calc_arclength
    match the training-time results (same scipy CubicSpline underneath)."""

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        psis: Optional[np.ndarray] = None,
        ks: Optional[np.ndarray] = None,
        vxs: Optional[np.ndarray] = None,
        axs: Optional[np.ndarray] = None,
    ):
        self.xs = xs
        self.ys = ys
        self.psis = psis
        self.ks = ks
        self.vxs = vxs
        self.axs = axs

        psis_spline = psis if psis is not None else np.zeros_like(xs)
        cosines_spline = np.cos(psis_spline)
        sines_spline = np.sin(psis_spline)
        ks_spline = ks if ks is not None else np.zeros_like(xs)
        vxs_spline = vxs if vxs is not None else np.zeros_like(xs)
        axs_spline = axs if axs is not None else np.zeros_like(xs)

        self.points = np.c_[
            self.xs, self.ys, cosines_spline, sines_spline,
            ks_spline, vxs_spline, axs_spline,
        ]
        if not np.all(self.points[-1] == self.points[0]):
            self.points = np.vstack((self.points, self.points[0]))

        self.s = self._calc_s(self.points[:, 0], self.points[:, 1])
        self.spline = interpolate.CubicSpline(self.s, self.points, bc_type="periodic")

    def _calc_s(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0.0]
        s.extend(np.cumsum(self.ds))
        return np.array(s)

    def find_segment_for_x(self, x: float):
        segment = np.searchsorted(self.spline.x, x, side="right") - 1
        return np.clip(segment, 0, len(self.spline.x) - 2)

    def predict_with_spline(self, point: float, segment, state_index: int = 0):
        exp_x = ((point - self.spline.x[segment]) ** np.arange(4)[::-1])[:, None]
        vec = self.spline.c[:, segment, state_index]
        return np.asarray(vec.dot(exp_x))

    def calc_position(self, s: float):
        segment = self.find_segment_for_x(s)
        x = self.predict_with_spline(s, segment, 0)[0]
        y = self.predict_with_spline(s, segment, 1)[0]
        return x, y

    def calc_yaw(self, s: float) -> float:
        if self.psis is None:
            dx, dy = self.spline(s, 1)[:2]
            return float(np.arctan2(dy, dx))
        segment = self.find_segment_for_x(s)
        cos = self.predict_with_spline(s, segment, 2)[0]
        sin = self.predict_with_spline(s, segment, 3)[0]
        return float(np.arctan2(sin, cos))

    def calc_curvature(self, s: float) -> float:
        if self.ks is None:
            dx, dy = self.spline(s, 1)[:2]
            ddx, ddy = self.spline(s, 2)[:2]
            return float((ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** 1.5))
        segment = self.find_segment_for_x(s)
        return float(self.predict_with_spline(s, segment, 4)[0])

    def calc_arclength(self, x: float, y: float) -> tuple[float, float]:
        """Global nearest-point projection -- same math/behavior as the
        training-time calc_arclength / calc_arclength_jax (s_guess is not
        actually used there either, it's a full search each call)."""
        ey, t, min_dist_segment = nearest_point_on_trajectory(
            np.array([x, y], dtype=np.float64), self.points[:, :2]
        )
        s = float(
            self.s[min_dist_segment]
            + t * (self.s[min_dist_segment + 1] - self.s[min_dist_segment])
        )
        return s, float(ey)


class Track:
    """Numpy/scipy-only Track: no jax, no network download. Loads centerline
    (+ optional raceline) from files already present on disk."""

    def __init__(self, xs, ys, filepath=None, centerline=None, raceline=None, s_frame_max=None):
        self.filepath = filepath
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if xs.shape != ys.shape:
            raise ValueError("inconsistent shapes for x, y")
        if xs.ndim != 1:
            raise ValueError("x and y must be 1-dimensional arrays.")
        _validate_reference_points(xs, ys)

        self.xs = xs
        self.ys = ys
        self.length = float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))
        self.s_frame_max = float(s_frame_max) if s_frame_max is not None else self.length

        self.centerline = centerline or CubicSplineND(xs, ys)
        self.raceline = raceline or self.centerline
        self.s_guess = 0.0

    @staticmethod
    def from_track_dir(track_dir) -> "Track":
        """Load a track that already exists locally -- no download, no PIL,
        no occupancy grid. Expects `<stem>_centerline.csv`
        ([x, y, w_left, w_right]) and optionally `<stem>_raceline.csv`
        ([s, x, y, psi, kappa, vx, ax]) inside track_dir, matching the
        f1tenth map format.
        """
        track_dir = pathlib.Path(track_dir)
        if not track_dir.exists():
            raise FileNotFoundError(f"track directory does not exist: {track_dir}")
        stem = track_dir.stem

        centerline_path = track_dir / f"{stem}_centerline.csv"
        if not centerline_path.exists():
            raise FileNotFoundError(f"centerline file not found: {centerline_path}")
        cl_data = np.loadtxt(centerline_path, delimiter=",")
        cl_data = _validate_waypoint_table(cl_data, "centerline", 4, "[x, y, w_left, w_right]")
        cl_xs, cl_ys = cl_data[:, 0], cl_data[:, 1]
        # cl_xs = cl_xs[::-1]
        # cl_ys = cl_ys[::-1]
        cl_psis = _calc_yaw_from_xy(cl_xs, cl_ys)
        cl_xs = np.append(cl_xs, cl_xs[0])
        cl_ys = np.append(cl_ys, cl_ys[0])
        cl_psis = np.append(cl_psis, cl_psis[0])
        centerline = CubicSplineND(cl_xs, cl_ys, cl_psis)

        raceline = None
        raceline_path = track_dir / f"{stem}_raceline.csv"
        if raceline_path.exists():
            rl_data = np.loadtxt(raceline_path, delimiter=";")
            rl_data = _validate_waypoint_table(
                rl_data, "raceline", 7, "[s, x, y, psi, kappa, vx, ax]"
            )
            xs, ys, psis, kappas, vxs, axs = (
                rl_data[:, 1], rl_data[:, 2], rl_data[:, 3],
                rl_data[:, 4], rl_data[:, 5], rl_data[:, 6],
            )
            raceline = CubicSplineND(xs, ys, psis, kappas, vxs, axs)

        return Track(xs=cl_xs, ys=cl_ys, centerline=centerline, raceline=raceline, filepath=track_dir)

    @staticmethod
    def from_centerline_file(centerline_path, raceline_path=None) -> "Track":
        """Load a track directly from a centerline CSV file path (and
        optionally a raceline CSV path), with no directory-naming
        convention required -- use this when the CSV just lives on disk
        wherever you put it (e.g. right next to your ROS node).

        centerline_path: path to a CSV with columns [x, y, w_left, w_right]
        raceline_path: optional path to a CSV with columns
            [s, x, y, psi, kappa, vx, ax] (';'-delimited)
        """
        centerline_path = pathlib.Path(centerline_path)
        if not centerline_path.exists():
            raise FileNotFoundError(f"centerline file not found: {centerline_path}")

        cl_data = np.loadtxt(centerline_path, delimiter=",")
        cl_data = _validate_waypoint_table(cl_data, "centerline", 4, "[x, y, w_left, w_right]")
        cl_xs, cl_ys = cl_data[:, 0], cl_data[:, 1]
        # cl_xs = cl_xs[::-1]
        # cl_ys = cl_ys[::-1]
        cl_psis = _calc_yaw_from_xy(cl_xs, cl_ys)
        cl_xs = np.append(cl_xs, cl_xs[0])
        cl_ys = np.append(cl_ys, cl_ys[0])
        cl_psis = np.append(cl_psis, cl_psis[0])
        centerline = CubicSplineND(cl_xs, cl_ys, cl_psis)

        raceline = None
        if raceline_path is not None:
            raceline_path = pathlib.Path(raceline_path)
            if not raceline_path.exists():
                raise FileNotFoundError(f"raceline file not found: {raceline_path}")
            rl_data = np.loadtxt(raceline_path, delimiter=";")
            rl_data = _validate_waypoint_table(
                rl_data, "raceline", 7, "[s, x, y, psi, kappa, vx, ax]"
            )
            xs, ys, psis, kappas, vxs, axs = (
                rl_data[:, 1], rl_data[:, 2], rl_data[:, 3],
                rl_data[:, 4], rl_data[:, 5], rl_data[:, 6],
            )
            raceline = CubicSplineND(xs, ys, psis, kappas, vxs, axs)

        return Track(
            xs=cl_xs, ys=cl_ys, centerline=centerline, raceline=raceline,
            filepath=centerline_path,
        )

    def cartesian_to_frenet(self, x: float, y: float, phi: float, s_guess=None):
        s, ey = self.centerline.calc_arclength(x, y)
        s = s % self.s_frame_max
        self.s_guess = s

        yaw = self.centerline.calc_yaw(s)
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        x_eval, y_eval = self.centerline.calc_position(s)
        dx = x - x_eval
        dy = y - y_eval
        distance_sign = np.sign(np.dot([dx, dy], normal))
        ey = ey * distance_sign

        phi = phi - yaw
        return s, ey, float(np.arctan2(np.sin(phi), np.cos(phi)))

    def frenet_to_cartesian(self, s: float, ey: float, ephi: float):
        s = s % self.s_frame_max
        x, y = self.centerline.calc_position(s)
        psi = self.centerline.calc_yaw(s)
        x -= ey * np.sin(psi)
        y += ey * np.cos(psi)
        psi += ephi
        return x, y, float(np.arctan2(np.sin(psi), np.cos(psi)))

    def curvature(self, s: float) -> float:
        s = s % self.s_frame_max
        return self.centerline.calc_curvature(s)

def lookahead_curvatures(track: "Track", s: float, offsets_m=(1.0, 2.0, 4.0)) -> np.ndarray:
    """Curvature at fixed lookahead distances (meters) ahead of s, wrapped
    around the track length. Matches training-time lookahead points of
    1m, 2m, 4m ahead of the vehicle's current arclength position."""
    s_points = (s + np.asarray(offsets_m)) % track.s_frame_max
    return np.array([track.curvature(sp) for sp in s_points], dtype=np.float64)