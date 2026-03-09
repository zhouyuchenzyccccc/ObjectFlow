"""Keep GPU busy by repeatedly running LIBERO regenerate script without keeping outputs.

This helper launches one or more workers that continuously execute:
    experiments/robot/libero/generate_dataset/regenerate_libero_dataset.py

Each worker writes to a temporary target directory and removes it after each run.
It also deletes the generated metainfo json, so no persistent dataset artifacts remain.

Example:
    python busy.py \
      --libero_task_suite libero_object \
      --libero_raw_data_dir /inspire/hdd/project/wuliqifa/public/dataset/libero/datasets/libero_object \
      --single_hdf5_name pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5 \
      --workers 2
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from multiprocessing import Event, Process
from pathlib import Path


def build_generate_cmd(args, temp_target_dir):
    cmd = [
        sys.executable,
        "experiments/robot/libero/generate_dataset/regenerate_libero_dataset.py",
        "--libero_task_suite",
        args.libero_task_suite,
        "--libero_raw_data_dir",
        args.libero_raw_data_dir,
        "--libero_target_dir",
        temp_target_dir,
        "--point_count",
        str(args.point_count),
        "--point_cube_size",
        str(args.point_cube_size),
        "--point_seed",
        str(args.point_seed),
        "--robot_point_weight",
        str(args.robot_point_weight),
        "--min_non_robot_ratio",
        str(args.min_non_robot_ratio),
        "--point_motion_threshold",
        str(args.point_motion_threshold),
        "--robot_active_ratio_threshold",
        str(args.robot_active_ratio_threshold),
        "--object_active_ratio_threshold",
        str(args.object_active_ratio_threshold),
        "--object_group_time_gap",
        str(args.object_group_time_gap),
        "--object_group_active_ratio_threshold",
        str(args.object_group_active_ratio_threshold),
    ]

    if args.single_hdf5_name:
        cmd.extend(["--single_hdf5_name", args.single_hdf5_name])

    return cmd


def cleanup_artifacts(temp_target_dir, metainfo_path):
    if os.path.isdir(temp_target_dir):
        shutil.rmtree(temp_target_dir, ignore_errors=True)
    if os.path.exists(metainfo_path):
        try:
            os.remove(metainfo_path)
        except OSError:
            pass


def worker_loop(worker_id, args, stop_event):
    metainfo_path = Path("experiments/robot/libero") / f"{args.libero_task_suite}_metainfo.json"

    while not stop_event.is_set():
        temp_target_dir = tempfile.mkdtemp(prefix=f"busy_worker{worker_id}_", dir=args.temp_root)
        cmd = build_generate_cmd(args, temp_target_dir)

        print(f"[worker {worker_id}] start run -> {temp_target_dir}")
        proc = subprocess.run(cmd, cwd=args.repo_root, text=True)
        print(f"[worker {worker_id}] run finished with code={proc.returncode}")

        cleanup_artifacts(temp_target_dir, str(Path(args.repo_root) / metainfo_path))

        if proc.returncode != 0 and args.stop_on_error:
            print(f"[worker {worker_id}] stop_on_error enabled, exiting.")
            stop_event.set()
            break

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".", help="Repository root where the generate script is run.")
    parser.add_argument("--libero_task_suite", type=str, required=True)
    parser.add_argument("--libero_raw_data_dir", type=str, required=True)
    parser.add_argument("--single_hdf5_name", type=str, default=None)

    parser.add_argument("--workers", type=int, default=1, help="Parallel workers to run generate script.")
    parser.add_argument("--temp_root", type=str, default=None, help="Parent directory for temporary output dirs.")
    parser.add_argument("--sleep_seconds", type=float, default=0.0, help="Sleep between runs per worker.")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop all workers if any run fails.")

    parser.add_argument("--point_count", type=int, default=1024)
    parser.add_argument("--point_cube_size", type=float, default=1.2)
    parser.add_argument("--point_seed", type=int, default=7)
    parser.add_argument("--robot_point_weight", type=float, default=0.2)
    parser.add_argument("--min_non_robot_ratio", type=float, default=0.7)
    parser.add_argument("--point_motion_threshold", type=float, default=2e-3)
    parser.add_argument("--robot_active_ratio_threshold", type=float, default=0.02)
    parser.add_argument("--object_active_ratio_threshold", type=float, default=0.01)
    parser.add_argument("--object_group_time_gap", type=int, default=8)
    parser.add_argument("--object_group_active_ratio_threshold", type=float, default=0.02)

    args = parser.parse_args()
    args.repo_root = str(Path(args.repo_root).resolve())

    stop_event = Event()
    workers = []

    def handle_signal(signum, frame):
        print("\n[busy] stopping workers...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"[busy] repo_root={args.repo_root}")
    print(f"[busy] workers={args.workers}, suite={args.libero_task_suite}")

    for worker_id in range(args.workers):
        p = Process(target=worker_loop, args=(worker_id, args, stop_event), daemon=False)
        p.start()
        workers.append(p)

    try:
        for p in workers:
            p.join()
    finally:
        stop_event.set()
        for p in workers:
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    main()
