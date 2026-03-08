"""Inspect and visualize point-track data in regenerated LIBERO HDF5 files.

This script validates that saved point displacement is consistent with absolute points:
    pointcloud_disp[t] == pointcloud_abs[t + 1] - pointcloud_abs[t]

It also exports quick visualizations for selected episodes.

Usage:
    python experiments/robot/libero/inspect_regenerated_libero_points.py \
        --dataset_dir /path/to/libero_object_no_noops \
        --output_dir /path/to/inspect_output
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _compute_metrics(point_abs: np.ndarray, point_disp: np.ndarray) -> Dict[str, float]:
    """Compute consistency metrics between saved displacement and recomputed displacement."""
    if point_abs.ndim != 3 or point_abs.shape[-1] != 3:
        raise ValueError(f"pointcloud_abs must be shaped (T, N, 3), got {point_abs.shape}")
    if point_disp.ndim != 3 or point_disp.shape[-1] != 3:
        raise ValueError(f"pointcloud_disp must be shaped (T-1, N, 3), got {point_disp.shape}")

    t_steps, n_points, _ = point_abs.shape
    expected_shape = (max(t_steps - 1, 0), n_points, 3)
    if point_disp.shape != expected_shape:
        raise ValueError(
            f"pointcloud_disp shape mismatch: expected {expected_shape}, got {point_disp.shape}"
        )

    if t_steps <= 1:
        return {
            "t_steps": float(t_steps),
            "n_points": float(n_points),
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "rmse": 0.0,
        }

    recomputed = point_abs[1:] - point_abs[:-1]
    diff = point_disp - recomputed
    abs_err = np.abs(diff)

    return {
        "t_steps": float(t_steps),
        "n_points": float(n_points),
        "max_abs_error": float(abs_err.max()),
        "mean_abs_error": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def _plot_episode(
    point_abs: np.ndarray,
    point_disp: np.ndarray,
    out_png: Path,
    title: str,
    sample_points: int,
) -> None:
    """Create a 3-panel visualization for an episode."""
    t_steps, n_points, _ = point_abs.shape
    point_count = min(sample_points, n_points)
    idx = np.arange(point_count)

    fig = plt.figure(figsize=(15, 4.8))

    # Panel 1: absolute points at first frame and last frame.
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    p0 = point_abs[0, idx]
    p1 = point_abs[-1, idx]
    ax1.scatter(p0[:, 0], p0[:, 1], p0[:, 2], s=6, alpha=0.8, label="t=0")
    ax1.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=6, alpha=0.8, label=f"t={t_steps - 1}")
    ax1.set_title("Abs Points: Start vs End")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(loc="best")

    # Panel 2: displacement magnitude over time.
    ax2 = fig.add_subplot(1, 3, 2)
    if point_disp.shape[0] > 0:
        disp_mag = np.linalg.norm(point_disp[:, idx], axis=-1)  # (T-1, K)
        mean_mag = disp_mag.mean(axis=1)
        max_mag = disp_mag.max(axis=1)
        ax2.plot(mean_mag, label="mean |disp|", linewidth=1.8)
        ax2.plot(max_mag, label="max |disp|", linewidth=1.2)
    ax2.set_title("Displacement Magnitude")
    ax2.set_xlabel("t")
    ax2.set_ylabel("meters")
    ax2.legend(loc="best")

    # Panel 3: error histogram for saved disp vs recomputed disp.
    ax3 = fig.add_subplot(1, 3, 3)
    if point_disp.shape[0] > 0:
        recomputed = point_abs[1:] - point_abs[:-1]
        err = np.linalg.norm((point_disp - recomputed).reshape(-1, 3), axis=1)
        ax3.hist(err, bins=40)
    ax3.set_title("|saved - recomputed| Histogram")
    ax3.set_xlabel("meters")
    ax3.set_ylabel("count")

    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _iter_hdf5_files(dataset_dir: Path) -> List[Path]:
    return sorted(dataset_dir.glob("*_demo.hdf5"))


def main(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = _iter_hdf5_files(dataset_dir)
    if not hdf5_files:
        raise FileNotFoundError(f"No '*_demo.hdf5' files found under: {dataset_dir}")

    summary_rows: List[Tuple[str, str, Dict[str, float]]] = []
    checked_eps = 0

    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                raise KeyError(f"Missing group 'data' in file: {hdf5_path}")

            for demo_key in sorted(f["data"].keys()):
                ep = f["data"][demo_key]
                if "obs" not in ep:
                    raise KeyError(f"Missing group 'obs' in {hdf5_path}:{demo_key}")

                obs = ep["obs"]
                required = [
                    "pointcloud_abs",
                    "pointcloud_disp",
                    "point_track_geom_ids",
                    "point_track_face_indices",
                    "point_track_face_vertex_indices",
                    "point_track_barycentric",
                ]
                for key in required:
                    if key not in obs:
                        raise KeyError(f"Missing dataset '{key}' in {hdf5_path}:{demo_key}/obs")

                point_abs = obs["pointcloud_abs"][()]
                point_disp = obs["pointcloud_disp"][()]
                metrics = _compute_metrics(point_abs, point_disp)
                summary_rows.append((hdf5_path.name, demo_key, metrics))
                checked_eps += 1

                if checked_eps <= args.num_visualize:
                    out_png = output_dir / "plots" / f"{hdf5_path.stem}_{demo_key}.png"
                    title = f"{hdf5_path.name} | {demo_key}"
                    _plot_episode(
                        point_abs=point_abs,
                        point_disp=point_disp,
                        out_png=out_png,
                        title=title,
                        sample_points=args.sample_points,
                    )

    # Save summary as csv-like txt for zero dependency.
    summary_path = output_dir / "summary_metrics.csv"
    with open(summary_path, "w", encoding="utf-8") as fw:
        fw.write("file,demo,t_steps,n_points,max_abs_error,mean_abs_error,rmse\n")
        for file_name, demo_key, m in summary_rows:
            fw.write(
                f"{file_name},{demo_key},{int(m['t_steps'])},{int(m['n_points'])},"
                f"{m['max_abs_error']:.8e},{m['mean_abs_error']:.8e},{m['rmse']:.8e}\n"
            )

    max_err_global = max(row[2]["max_abs_error"] for row in summary_rows)
    mean_rmse_global = float(np.mean([row[2]["rmse"] for row in summary_rows]))

    print(f"Checked episodes: {checked_eps}")
    print(f"Summary saved to: {summary_path}")
    print(f"Plots saved under: {output_dir / 'plots'}")
    print(f"Global max abs error: {max_err_global:.8e}")
    print(f"Global mean RMSE: {mean_rmse_global:.8e}")

    if max_err_global > args.error_tol:
        print(
            f"[WARN] max_abs_error ({max_err_global:.8e}) > error_tol ({args.error_tol:.8e}). "
            "Please inspect affected episodes in summary_metrics.csv."
        )
    else:
        print(f"[OK] Displacement consistency check passed with tol={args.error_tol:.8e}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing regenerated *_demo.hdf5 files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for validation summary and figures.")
    parser.add_argument("--num_visualize", type=int, default=8, help="Number of episodes to visualize.")
    parser.add_argument("--sample_points", type=int, default=512, help="How many points to draw in plots per episode.")
    parser.add_argument(
        "--error_tol",
        type=float,
        default=1e-6,
        help="Tolerance for max absolute displacement error (meters).",
    )
    args = parser.parse_args()
    main(args)
