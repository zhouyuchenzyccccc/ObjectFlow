"""Render point flow directly on the original saved point cloud.

This script does NOT resample points by default. It visualizes flow vectors directly on
`pointcloud_abs` / `pointcloud_disp` stored in regenerated LIBERO HDF5 files.

Usage:
    python experiments/robot/libero/generate_dataset/visualize_libero_pointflow_on_original_cloud.py \
        --dataset_dir /path/to/libero_object_no_noops \
        --output_video /path/to/original_cloud_flow.mp4
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def _list_hdf5_files(dataset_dir: Path) -> List[Path]:
    return sorted(dataset_dir.glob("*_demo.hdf5"))


def _collect_demo_refs(hdf5_files: List[Path]) -> List[Tuple[Path, str]]:
    refs: List[Tuple[Path, str]] = []
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                continue
            for demo_key in sorted(f["data"].keys()):
                ep = f["data"][demo_key]
                if "obs" not in ep:
                    continue
                obs = ep["obs"]
                if "pointcloud_abs" in obs and "pointcloud_disp" in obs:
                    refs.append((hdf5_path, demo_key))
    return refs


def _load_demo_data(hdf5_path: Path, demo_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        obs = f["data"][demo_key]["obs"]
        point_abs = obs["pointcloud_abs"][()].astype(np.float32)  # (T, N, 3)
        point_disp = obs["pointcloud_disp"][()].astype(np.float32)  # (T-1, N, 3)
        is_robot = (
            obs["point_track_is_robot"][()].astype(bool)
            if "point_track_is_robot" in obs
            else np.zeros((point_abs.shape[1],), dtype=bool)
        )
    return point_abs, point_disp, is_robot


def _set_axes_equal(ax: plt.Axes, xyz_min: np.ndarray, xyz_max: np.ndarray) -> None:
    center = (xyz_min + xyz_max) / 2.0
    extent = np.max(xyz_max - xyz_min)
    extent = max(float(extent), 1e-6)
    half = extent / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _render_frames(
    point_abs: np.ndarray,
    point_disp: np.ndarray,
    is_robot: np.ndarray,
    arrow_stride: int,
    elev: float,
    azim: float,
) -> List[np.ndarray]:
    t_steps, n_points, _ = point_abs.shape

    xyz_min = point_abs.reshape(-1, 3).min(axis=0)
    xyz_max = point_abs.reshape(-1, 3).max(axis=0)

    fig = plt.figure(figsize=(8.6, 7.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    frames: List[np.ndarray] = []
    robot_mask = is_robot.astype(bool)
    object_mask = ~robot_mask

    for t in range(t_steps):
        ax.cla()
        pts = point_abs[t]

        if np.any(object_mask):
            ax.scatter(
                pts[object_mask, 0],
                pts[object_mask, 1],
                pts[object_mask, 2],
                s=6,
                alpha=0.65,
                color="deepskyblue",
                label="object points",
            )
        if np.any(robot_mask):
            ax.scatter(
                pts[robot_mask, 0],
                pts[robot_mask, 1],
                pts[robot_mask, 2],
                s=6,
                alpha=0.85,
                color="orangered",
                label="robot points",
            )

        if t < point_disp.shape[0] and arrow_stride > 0:
            arrow_idx = np.arange(0, n_points, arrow_stride)
            p = pts[arrow_idx]
            d = point_disp[t, arrow_idx]
            ax.quiver(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                d[:, 0],
                d[:, 1],
                d[:, 2],
                length=1.0,
                normalize=False,
                color="crimson",
                linewidth=0.6,
            )

        _set_axes_equal(ax, xyz_min, xyz_max)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"Original Saved Point Cloud + Flow | frame {t + 1}/{t_steps} | N={n_points}")

        if t == 0:
            ax.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)
        frames.append(frame[:, :, :3].copy())

    plt.close(fig)
    return frames


def _write_video(frames: List[np.ndarray], output_video: Path, fps: int) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_video.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(output_video, frames, fps=fps)
        return

    try:
        with imageio.get_writer(output_video, fps=fps, codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)
    except Exception as exc:
        fallback = output_video.with_suffix(".gif")
        imageio.mimsave(fallback, frames, fps=fps)
        raise RuntimeError(
            f"Failed to write '{output_video}' as mp4 ({exc}). Saved gif fallback to '{fallback}'."
        ) from exc


def main(args: argparse.Namespace) -> None:
    if args.hdf5_path:
        hdf5_path = Path(args.hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file does not exist: {hdf5_path}")
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                raise KeyError(f"Missing 'data' in file: {hdf5_path}")
            demo_keys = sorted(f["data"].keys())
        if not demo_keys:
            raise RuntimeError(f"No demos found in: {hdf5_path}")
        demo_key = args.demo_key if args.demo_key else demo_keys[0]
        if demo_key not in demo_keys:
            raise KeyError(f"demo_key '{demo_key}' not in {hdf5_path.name}, available: {demo_keys[:5]}...")
    else:
        dataset_dir = Path(args.dataset_dir)
        hdf5_files = _list_hdf5_files(dataset_dir)
        if not hdf5_files:
            raise FileNotFoundError(f"No '*_demo.hdf5' files found in: {dataset_dir}")
        demo_refs = _collect_demo_refs(hdf5_files)
        if not demo_refs:
            raise RuntimeError("No demos with pointcloud_abs/pointcloud_disp were found.")
        rng = np.random.default_rng(args.seed)
        choice_idx = int(rng.integers(len(demo_refs)))
        hdf5_path, demo_key = demo_refs[choice_idx]

    point_abs, point_disp, is_robot = _load_demo_data(hdf5_path, demo_key)

    # Optional cap for rendering speed; default is no resampling.
    if args.max_render_points > 0 and args.max_render_points < point_abs.shape[1]:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(point_abs.shape[1], size=args.max_render_points, replace=False))
        point_abs = point_abs[:, idx, :]
        point_disp = point_disp[:, idx, :] if point_disp.shape[0] > 0 else point_disp
        is_robot = is_robot[idx]

    print(f"Selected demo: {hdf5_path.name}:{demo_key}")
    print(f"Using original saved pointcloud_abs/disp with shape abs={point_abs.shape}, disp={point_disp.shape}")

    frames = _render_frames(
        point_abs=point_abs,
        point_disp=point_disp,
        is_robot=is_robot,
        arrow_stride=args.arrow_stride,
        elev=args.elev,
        azim=args.azim,
    )

    output_video = Path(args.output_video)
    _write_video(frames, output_video, fps=args.fps)
    print(f"Saved video: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory containing regenerated *_demo.hdf5 files.")
    parser.add_argument("--hdf5_path", type=str, default=None, help="Direct path to a single *_demo.hdf5 file.")
    parser.add_argument("--demo_key", type=str, default=None, help="Demo key in HDF5, e.g. demo_0.")
    parser.add_argument("--output_video", type=str, required=True, help="Output video path, e.g. /tmp/original_cloud_flow.mp4")
    parser.add_argument("--seed", type=int, default=7, help="Seed for random demo selection.")
    parser.add_argument("--max_render_points", type=int, default=-1, help="Optional render cap; <=0 means render all saved points.")
    parser.add_argument("--arrow_stride", type=int, default=8, help="Draw one flow arrow every N points; <=0 disables arrows.")
    parser.add_argument("--fps", type=int, default=12, help="Video FPS.")
    parser.add_argument("--elev", type=float, default=22.0, help="3D camera elevation angle.")
    parser.add_argument("--azim", type=float, default=-55.0, help="3D camera azimuth angle.")
    args = parser.parse_args()

    if args.dataset_dir is None and args.hdf5_path is None:
        raise ValueError("Provide either --dataset_dir or --hdf5_path")
    if args.dataset_dir is not None and args.hdf5_path is not None:
        raise ValueError("Use only one of --dataset_dir and --hdf5_path")

    main(args)
