"""Render a random regenerated LIBERO demo as a point-flow video.

The script loads one random demo from regenerated LIBERO HDF5 files and writes a video
showing how the scene point cloud evolves over time.

Usage:
    python experiments/robot/libero/generate_dataset/visualize_libero_pointflow_video.py \
        --dataset_dir /path/to/libero_object_no_noops \
        --output_video /path/to/random_demo_pointflow.mp4 \
        --trail_stride 16 \
        --trail_len 25
"""

import argparse
from pathlib import Path
from typing import List, Tuple

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


def _load_demo_points(hdf5_path: Path, demo_key: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        obs = f["data"][demo_key]["obs"]
        point_abs = obs["pointcloud_abs"][()].astype(np.float32)  # (T, N, 3)
        point_disp = obs["pointcloud_disp"][()].astype(np.float32)  # (T-1, N, 3)
    return point_abs, point_disp


def _sample_points(
    point_abs: np.ndarray,
    point_disp: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    t_steps, n_points, _ = point_abs.shape
    if max_points >= n_points:
        return point_abs, point_disp

    idx = rng.choice(n_points, size=max_points, replace=False)
    idx = np.sort(idx)
    sampled_abs = point_abs[:, idx, :]
    sampled_disp = point_disp[:, idx, :] if point_disp.shape[0] > 0 else point_disp
    return sampled_abs, sampled_disp


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
    arrow_stride: int,
    trail_stride: int,
    trail_len: int,
    trail_alpha: float,
    trail_width: float,
    elev: float,
    azim: float,
) -> List[np.ndarray]:
    t_steps, n_points, _ = point_abs.shape

    xyz_min = point_abs.reshape(-1, 3).min(axis=0)
    xyz_max = point_abs.reshape(-1, 3).max(axis=0)

    fig = plt.figure(figsize=(8.4, 7.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    frames: List[np.ndarray] = []
    for t in range(t_steps):
        ax.cla()

        pts = point_abs[t]
        if t < point_disp.shape[0]:
            speed = np.linalg.norm(point_disp[t], axis=1)
        else:
            speed = np.zeros((n_points,), dtype=np.float32)

        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=speed,
            cmap="viridis",
            s=6,
            vmin=0.0,
            vmax=max(float(np.percentile(speed, 99)), 1e-6),
            alpha=0.95,
        )

        if t < point_disp.shape[0] and arrow_stride > 0:
            arrow_idx = np.arange(0, n_points, arrow_stride)
            p = pts[arrow_idx]
            d = point_disp[t, arrow_idx]
            ax.quiver(
                p[:, 0], p[:, 1], p[:, 2],
                d[:, 0], d[:, 1], d[:, 2],
                length=1.0,
                normalize=False,
                color="crimson",
                linewidth=0.6,
            )

        # Draw 3D trajectory trails for a sparse subset of points, similar to 2D optical-flow tracks.
        if trail_len > 1 and trail_stride > 0:
            trail_idx = np.arange(0, n_points, trail_stride)
            start = max(0, t - trail_len + 1)
            traj = point_abs[start : t + 1, trail_idx, :]  # (Lt, Nt, 3)
            for k in range(traj.shape[1]):
                path = traj[:, k, :]
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    path[:, 2],
                    color="deepskyblue",
                    alpha=trail_alpha,
                    linewidth=trail_width,
                )

        _set_axes_equal(ax, xyz_min, xyz_max)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"Scene Point Flow | frame {t + 1}/{t_steps}")

        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
        cbar.set_label("|disp| (m)")

        fig.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)
        frames.append(frame[:, :, :3].copy())

        cbar.remove()

    plt.close(fig)
    return frames


def _write_video(frames: List[np.ndarray], output_video: Path, fps: int) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_video.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(output_video, frames, fps=fps)
        return

    # For mp4, imageio uses ffmpeg backend if available.
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
    dataset_dir = Path(args.dataset_dir)
    output_video = Path(args.output_video)

    hdf5_files = _list_hdf5_files(dataset_dir)
    if not hdf5_files:
        raise FileNotFoundError(f"No '*_demo.hdf5' files found in: {dataset_dir}")

    demo_refs = _collect_demo_refs(hdf5_files)
    if not demo_refs:
        raise RuntimeError("No demos with pointcloud_abs/pointcloud_disp were found.")

    rng = np.random.default_rng(args.seed)
    choice_idx = int(rng.integers(len(demo_refs)))
    hdf5_path, demo_key = demo_refs[choice_idx]

    point_abs, point_disp = _load_demo_points(hdf5_path, demo_key)
    point_abs, point_disp = _sample_points(point_abs, point_disp, args.max_points, rng)

    print(f"Selected demo: {hdf5_path.name}:{demo_key}")
    print(f"Pointcloud shape: abs={point_abs.shape}, disp={point_disp.shape}")

    frames = _render_frames(
        point_abs=point_abs,
        point_disp=point_disp,
        arrow_stride=args.arrow_stride,
        trail_stride=args.trail_stride,
        trail_len=args.trail_len,
        trail_alpha=args.trail_alpha,
        trail_width=args.trail_width,
        elev=args.elev,
        azim=args.azim,
    )

    _write_video(frames, output_video, fps=args.fps)
    print(f"Saved video: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing regenerated *_demo.hdf5 files.")
    parser.add_argument("--output_video", type=str, required=True, help="Output path, e.g. /tmp/demo.mp4 or /tmp/demo.gif")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for selecting one demo.")
    parser.add_argument("--max_points", type=int, default=1024, help="Maximum number of points rendered per frame.")
    parser.add_argument("--arrow_stride", type=int, default=24, help="Draw one flow arrow every N points.")
    parser.add_argument("--trail_stride", type=int, default=16, help="Draw one trajectory line every N points.")
    parser.add_argument("--trail_len", type=int, default=25, help="Number of past frames shown as trajectory trail.")
    parser.add_argument("--trail_alpha", type=float, default=0.35, help="Transparency of trajectory trails.")
    parser.add_argument("--trail_width", type=float, default=0.8, help="Line width for trajectory trails.")
    parser.add_argument("--fps", type=int, default=12, help="Video frame rate.")
    parser.add_argument("--elev", type=float, default=22.0, help="3D camera elevation angle.")
    parser.add_argument("--azim", type=float, default=-55.0, help="3D camera azimuth angle.")
    args = parser.parse_args()
    main(args)
