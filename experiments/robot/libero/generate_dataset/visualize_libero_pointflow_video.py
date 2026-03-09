"""Render a random regenerated LIBERO demo as point-flow and 2D videos.

The script loads one random demo from regenerated LIBERO HDF5 files and writes a video
showing how the scene point cloud evolves over time.

Usage:
    python experiments/robot/libero/generate_dataset/visualize_libero_pointflow_video.py \
        --dataset_dir /path/to/libero_object_no_noops \
        --output_video /path/to/random_demo_pointflow.mp4 \
        --output_2d_video /path/to/random_demo_agentview.mp4 \
        --trail_stride 16 \
        --trail_len 25
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PHASE_ID_TO_NAME = {
    0: "idle",
    1: "robot_move",
    2: "grasp_or_contact",
    3: "co_move",
    4: "object_switch",
    5: "object_move",
}


def _list_hdf5_files(dataset_dir: Path) -> List[Path]:
    return sorted(dataset_dir.glob("*_demo.hdf5"))


def _collect_demo_refs(hdf5_files: List[Path], rgb_key: str) -> List[Tuple[Path, str]]:
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
                if "pointcloud_abs" in obs and "pointcloud_disp" in obs and rgb_key in obs:
                    refs.append((hdf5_path, demo_key))
    return refs


def _load_demo_data(
    hdf5_path: Path,
    demo_key: str,
    rgb_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        obs = f["data"][demo_key]["obs"]
        point_abs = obs["pointcloud_abs"][()].astype(np.float32)  # (T, N, 3)
        point_disp = obs["pointcloud_disp"][()].astype(np.float32)  # (T-1, N, 3)
        rgb_frames = obs[rgb_key][()]  # (T, H, W, 3)

        n_steps = point_disp.shape[0]
        n_points = point_abs.shape[1]
        is_robot_point = obs["point_track_is_robot"][()].astype(bool) if "point_track_is_robot" in obs else np.zeros((n_points,), dtype=bool)
        object_group_id = (
            obs["point_motion_object_group_id"][()].astype(np.int32)
            if "point_motion_object_group_id" in obs
            else np.full((n_points,), -1, dtype=np.int32)
        )
        point_is_moving = (
            obs["point_motion_is_moving"][()].astype(np.uint8)
            if "point_motion_is_moving" in obs
            else (np.linalg.norm(point_disp, axis=-1) > 1e-3).astype(np.uint8)
        )
        phase_label = obs["phase_label"][()].astype(np.uint8) if "phase_label" in obs else np.zeros((n_steps,), dtype=np.uint8)
        phase_dominant_group = (
            obs["phase_dominant_object_group"][()].astype(np.int32)
            if "phase_dominant_object_group" in obs
            else np.full((n_steps,), -1, dtype=np.int32)
        )
        if "point_motion_object_label" in obs:
            object_labels = np.asarray(obs["point_motion_object_label"][()]).astype("S32")
        else:
            object_labels = np.full((n_points,), b"", dtype="S32")

    return (
        point_abs,
        point_disp,
        rgb_frames,
        is_robot_point,
        object_group_id,
        point_is_moving,
        phase_label,
        phase_dominant_group,
        object_labels,
    )


def _group_id_to_name(group_id: int) -> str:
    if group_id < 0:
        return "robot"
    if group_id == 0:
        return "background"

    idx = group_id - 1
    chars = []
    n = idx + 1
    while n > 0:
        n -= 1
        chars.append(chr(ord("a") + (n % 26)))
        n //= 26
    return f"move_object_{''.join(reversed(chars))}"


def _normalize_rgb_frames(rgb_frames: np.ndarray) -> np.ndarray:
    """Convert RGB frames to uint8, shape (T, H, W, 3)."""
    if rgb_frames.ndim != 4 or rgb_frames.shape[-1] != 3:
        raise ValueError(f"RGB frames must be shaped (T, H, W, 3), got {rgb_frames.shape}")

    if rgb_frames.dtype == np.uint8:
        return rgb_frames

    if np.issubdtype(rgb_frames.dtype, np.floating):
        # Handle float images in either [0, 1] or [0, 255].
        max_val = float(np.max(rgb_frames)) if rgb_frames.size > 0 else 1.0
        scale = 255.0 if max_val <= 1.0 else 1.0
        return np.clip(rgb_frames * scale, 0, 255).astype(np.uint8)

    return np.clip(rgb_frames, 0, 255).astype(np.uint8)


def _maybe_rotate_rgb_frames(rgb_frames: np.ndarray, rotate_180: bool) -> np.ndarray:
    """Rotate RGB frames by 180 degrees when environment camera is upside down."""
    if not rotate_180:
        return rgb_frames
    # Rotate across spatial axes H/W, keep time/channel axes unchanged.
    return np.rot90(rgb_frames, k=2, axes=(1, 2)).copy()


def _sample_points(
    point_abs: np.ndarray,
    point_disp: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_steps, n_points, _ = point_abs.shape
    if max_points >= n_points:
        return point_abs, point_disp, np.arange(n_points, dtype=np.int32)

    idx = rng.choice(n_points, size=max_points, replace=False)
    idx = np.sort(idx)
    sampled_abs = point_abs[:, idx, :]
    sampled_disp = point_disp[:, idx, :] if point_disp.shape[0] > 0 else point_disp
    return sampled_abs, sampled_disp, idx.astype(np.int32)


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
    is_robot_point: np.ndarray,
    object_group_id: np.ndarray,
    point_is_moving: np.ndarray,
    phase_label: np.ndarray,
    phase_dominant_group: np.ndarray,
    object_labels: np.ndarray,
    arrow_stride: int,
    trail_stride: int,
    trail_len: int,
    trail_alpha: float,
    trail_width: float,
    elev: float,
    azim: float,
) -> List[np.ndarray]:
    t_steps, n_points, _ = point_abs.shape
    group_cmap = plt.get_cmap("tab10")
    robot_color = "orangered"
    unknown_object_color = "steelblue"
    moving_highlight_color = "yellow"

    xyz_min = point_abs.reshape(-1, 3).min(axis=0)
    xyz_max = point_abs.reshape(-1, 3).max(axis=0)

    fig = plt.figure(figsize=(10.0, 7.2), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.7], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax_info = fig.add_subplot(gs[0, 1])

    object_groups_all = [int(g) for g in np.unique(object_group_id[object_group_id >= 0])]
    group_color_map = {gid: group_cmap(gid % 10) for gid in object_groups_all}

    frames: List[np.ndarray] = []
    for t in range(t_steps):
        ax.cla()
        ax_info.cla()
        ax_info.set_axis_off()

        pts = point_abs[t]
        if t < point_is_moving.shape[0]:
            moving_mask = point_is_moving[t].astype(bool)
        else:
            moving_mask = np.zeros((n_points,), dtype=bool)

        robot_mask = is_robot_point.astype(bool)
        object_mask = ~robot_mask

        # Robot points (single semantic class).
        if np.any(robot_mask):
            ax.scatter(
                pts[robot_mask, 0],
                pts[robot_mask, 1],
                pts[robot_mask, 2],
                color=robot_color,
                s=8,
                alpha=0.8,
                label="robot points" if t == 0 else None,
            )

        # Object points are grouped as background (0) or moved objects (>=1).
        object_groups = np.unique(object_group_id[object_mask & (object_group_id >= 0)])
        for group_id in object_groups:
            group_mask = object_mask & (object_group_id == group_id)
            color = group_color_map[int(group_id)]
            ax.scatter(
                pts[group_mask, 0],
                pts[group_mask, 1],
                pts[group_mask, 2],
                color=color,
                s=7,
                alpha=0.65,
                label=f"{_group_id_to_name(int(group_id))}" if t == 0 else None,
            )

        unknown_object_mask = object_mask & (object_group_id < 0)
        if np.any(unknown_object_mask):
            ax.scatter(
                pts[unknown_object_mask, 0],
                pts[unknown_object_mask, 1],
                pts[unknown_object_mask, 2],
                color=unknown_object_color,
                s=7,
                alpha=0.45,
                label="object (unclustered)" if t == 0 else None,
            )

        # Highlight currently moving points with bright rings.
        if np.any(moving_mask):
            ax.scatter(
                pts[moving_mask, 0],
                pts[moving_mask, 1],
                pts[moving_mask, 2],
                facecolors="none",
                edgecolors=moving_highlight_color,
                linewidths=0.5,
                s=18,
                alpha=0.95,
                label="moving points" if t == 0 else None,
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
                point_id = trail_idx[k]
                if is_robot_point[point_id]:
                    line_color = "orange"
                else:
                    gid = int(object_group_id[point_id])
                    line_color = group_color_map[gid] if gid >= 0 else "deepskyblue"
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    path[:, 2],
                    color=line_color,
                    alpha=trail_alpha,
                    linewidth=trail_width,
                )

        _set_axes_equal(ax, xyz_min, xyz_max)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        phase_idx = t if t < phase_label.shape[0] else max(phase_label.shape[0] - 1, 0)
        phase_id = int(phase_label[phase_idx]) if phase_label.shape[0] > 0 else 0
        phase_name = PHASE_ID_TO_NAME.get(phase_id, f"unknown_{phase_id}")
        dominant_group = int(phase_dominant_group[phase_idx]) if phase_dominant_group.shape[0] > 0 else -1
        dominant_group_name = _group_id_to_name(dominant_group)
        ax.set_title(
            "Scene Point Flow | "
            f"frame {t + 1}/{t_steps} | phase={phase_name}({phase_id}) | "
            f"dom={dominant_group_name}({dominant_group})"
        )

        # Persistent side panel makes semantic color coding explicit in every frame.
        y = 0.98
        line_h = 0.08
        ax_info.text(0.02, y, "Legend", fontsize=11, fontweight="bold", va="top")
        y -= line_h
        ax_info.text(0.02, y, f"frame: {t + 1}/{t_steps}", fontsize=9, va="top")
        y -= line_h * 0.85
        ax_info.text(0.02, y, f"phase: {phase_name} ({phase_id})", fontsize=9, va="top")
        y -= line_h * 0.85
        ax_info.text(0.02, y, f"dom object: {dominant_group_name} ({dominant_group})", fontsize=9, va="top")
        y -= line_h

        ax_info.text(0.02, y, "Semantic Colors", fontsize=9.5, fontweight="bold", va="top")
        y -= line_h * 0.8

        def _legend_row(label, color):
            nonlocal y
            ax_info.add_patch(
                mpatches.Rectangle(
                    (0.02, y - 0.028),
                    0.075,
                    0.03,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.3,
                    transform=ax_info.transAxes,
                    clip_on=False,
                )
            )
            ax_info.text(0.11, y - 0.004, label, fontsize=8.6, va="center")
            y -= line_h * 0.68

        _legend_row("robot", robot_color)
        _legend_row("moving highlight", moving_highlight_color)
        _legend_row("object (legacy unclustered)", unknown_object_color)

        if object_groups_all:
            ax_info.text(0.02, y, "Object Labels", fontsize=9.5, fontweight="bold", va="top")
            y -= line_h * 0.8
            for gid in object_groups_all:
                label_name = _group_id_to_name(gid)
                if object_labels.size == object_group_id.size and gid >= 0:
                    mask = object_group_id == gid
                    if np.any(mask):
                        raw = object_labels[np.where(mask)[0][0]]
                        try:
                            decoded = raw.decode("ascii") if isinstance(raw, (bytes, np.bytes_)) else str(raw)
                            if decoded:
                                label_name = decoded
                        except Exception:
                            pass
                _legend_row(f"{label_name} ({gid})", group_color_map[gid])

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


def _resize_nearest(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image with nearest-neighbor interpolation using numpy only."""
    src_h, src_w = frame.shape[:2]
    if src_h == target_h and src_w == target_w:
        return frame

    row_idx = np.linspace(0, src_h - 1, target_h).astype(np.int32)
    col_idx = np.linspace(0, src_w - 1, target_w).astype(np.int32)
    return frame[row_idx][:, col_idx]


def _build_side_by_side_frames(pointflow_frames: List[np.ndarray], rgb_frames: np.ndarray) -> List[np.ndarray]:
    """Concatenate point-flow frame and 2D observation frame horizontally."""
    if len(pointflow_frames) == 0:
        return []

    t_steps = min(len(pointflow_frames), rgb_frames.shape[0])
    comparison_frames: List[np.ndarray] = []
    for t in range(t_steps):
        left = pointflow_frames[t]
        right = rgb_frames[t]
        right_resized = _resize_nearest(right, left.shape[0], left.shape[1])
        merged = np.concatenate([left, right_resized], axis=1)
        comparison_frames.append(merged)

    return comparison_frames


def main(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_video = Path(args.output_video)
    output_2d_video = Path(args.output_2d_video) if args.output_2d_video else output_video.with_name(
        f"{output_video.stem}_2d{output_video.suffix}"
    )
    output_comparison_video: Optional[Path]
    if args.output_comparison_video:
        output_comparison_video = Path(args.output_comparison_video)
    elif args.save_comparison:
        output_comparison_video = output_video.with_name(f"{output_video.stem}_compare{output_video.suffix}")
    else:
        output_comparison_video = None

    hdf5_files = _list_hdf5_files(dataset_dir)
    if not hdf5_files:
        raise FileNotFoundError(f"No '*_demo.hdf5' files found in: {dataset_dir}")

    demo_refs = _collect_demo_refs(hdf5_files, args.rgb_key)
    if not demo_refs:
        raise RuntimeError(
            f"No demos with pointcloud_abs/pointcloud_disp and rgb key '{args.rgb_key}' were found."
        )

    rng = np.random.default_rng(args.seed)
    choice_idx = int(rng.integers(len(demo_refs)))
    hdf5_path, demo_key = demo_refs[choice_idx]

    (
        point_abs,
        point_disp,
        rgb_frames,
        is_robot_point,
        object_group_id,
        point_is_moving,
        phase_label,
        phase_dominant_group,
        object_labels,
    ) = _load_demo_data(hdf5_path, demo_key, args.rgb_key)
    rgb_frames = _normalize_rgb_frames(rgb_frames)
    rgb_frames = _maybe_rotate_rgb_frames(rgb_frames, args.rotate_rgb_180)
    point_abs, point_disp, subset_idx = _sample_points(point_abs, point_disp, args.max_points, rng)
    if subset_idx.shape[0] != is_robot_point.shape[0]:
        is_robot_point = is_robot_point[subset_idx]
        object_group_id = object_group_id[subset_idx]
        object_labels = object_labels[subset_idx]
        if point_is_moving.shape[1] >= subset_idx.shape[0]:
            point_is_moving = point_is_moving[:, subset_idx]

    print(f"Selected demo: {hdf5_path.name}:{demo_key}")
    print(f"Pointcloud shape: abs={point_abs.shape}, disp={point_disp.shape}")
    print(f"RGB shape ({args.rgb_key}): {rgb_frames.shape}, rotated_180={args.rotate_rgb_180}")

    frames = _render_frames(
        point_abs=point_abs,
        point_disp=point_disp,
        is_robot_point=is_robot_point,
        object_group_id=object_group_id,
        point_is_moving=point_is_moving,
        phase_label=phase_label,
        phase_dominant_group=phase_dominant_group,
        object_labels=object_labels,
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

    # Save corresponding LIBERO 2D demo frames for direct comparison.
    rgb_list = [rgb_frames[t] for t in range(rgb_frames.shape[0])]
    _write_video(rgb_list, output_2d_video, fps=args.fps)
    print(f"Saved 2D video ({args.rgb_key}): {output_2d_video}")

    if output_comparison_video is not None:
        comparison_frames = _build_side_by_side_frames(frames, rgb_frames)
        _write_video(comparison_frames, output_comparison_video, fps=args.fps)
        print(f"Saved comparison video (3D|2D): {output_comparison_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing regenerated *_demo.hdf5 files.")
    parser.add_argument("--output_video", type=str, required=True, help="Output path, e.g. /tmp/demo.mp4 or /tmp/demo.gif")
    parser.add_argument(
        "--output_2d_video",
        type=str,
        default=None,
        help="Output path for LIBERO 2D demo video. Default: <output_video_stem>_2d.<ext>",
    )
    parser.add_argument(
        "--output_comparison_video",
        type=str,
        default=None,
        help="Output path for side-by-side comparison video. Used when --save_comparison is set.",
    )
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="If set, also save side-by-side comparison video (left=3D point-flow, right=2D RGB).",
    )
    parser.add_argument(
        "--rgb_key",
        type=str,
        default="agentview_rgb",
        choices=["agentview_rgb", "eye_in_hand_rgb"],
        help="Which 2D observation stream to save from HDF5.",
    )
    parser.add_argument(
        "--rotate_rgb_180",
        action="store_true",
        default=True,
        help="Rotate RGB frames by 180 degrees before saving (default: enabled).",
    )
    parser.add_argument(
        "--no_rotate_rgb_180",
        action="store_false",
        dest="rotate_rgb_180",
        help="Disable 180-degree RGB rotation.",
    )
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
