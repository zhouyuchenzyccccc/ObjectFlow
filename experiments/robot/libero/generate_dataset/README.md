# LIBERO Dataset Generation Tools

This folder contains scripts for regenerating LIBERO HDF5 data, validating saved point-flow fields, and visualizing scene point cloud dynamics.

## Files

- `regenerate_libero_dataset.py`: Replay raw LIBERO demos and regenerate filtered HDF5 dataset with point tracks.
- `inspect_regenerated_libero_points.py`: Validate `pointcloud_abs` / `pointcloud_disp` consistency and export diagnostic plots.
- `visualize_libero_pointflow_video.py`: Randomly pick one complete demo and render scene point-flow as a video.

## 0. Environment Preparation

Use your `openvla-oft` environment and make sure LIBERO-related paths are visible:

```bash
export PYTHONPATH=$PYTHONPATH:/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/LIBERO
export PYTHONPATH=$PYTHONPATH:/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/diffusion_policy
```

Recommended dependencies for these tools:

- `h5py`
- `numpy`
- `matplotlib`
- `imageio`
- `mujoco`
- `robosuite`
- `libero`

## 1. Regenerate Dataset

### Purpose

Replay original LIBERO demonstrations in simulator, then save regenerated trajectories while:

- filtering no-op actions,
- keeping successful demos only,
- saving Pri4R-style point tracks (`pointcloud_abs`, `pointcloud_disp`, etc.).

### Command

```bash
python experiments/robot/libero/generate_dataset/regenerate_libero_dataset.py \
  --libero_task_suite libero_object \
  --libero_raw_data_dir /inspire/hdd/project/wuliqifa/public/dataset/libero/datasets/libero_object \
  --libero_target_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/ObjectFlow/datasets/libero_object_no_noops \
  --point_count 1024 \
  --point_cube_size 1.2 \
  --point_seed 7
```

### Key Arguments

- `--libero_task_suite`: one of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`
- `--libero_raw_data_dir`: raw LIBERO HDF5 folder
- `--libero_target_dir`: output folder for regenerated HDF5
- `--point_count`: number of sampled tracked points per episode
- `--point_cube_size`: robot-centered crop cube size in meters
- `--point_seed`: random seed for point sampling

### Output

- Regenerated task files: `<libero_target_dir>/*_demo.hdf5`
- Metainfo JSON: `experiments/robot/libero/<task_suite>_metainfo.json`

## 2. Validate Saved Point Flow

### Purpose

Check whether saved point displacement is numerically consistent:

- `pointcloud_disp[t] == pointcloud_abs[t + 1] - pointcloud_abs[t]`

Also generate quick per-episode diagnostics.

### Command

```bash
python experiments/robot/libero/generate_dataset/inspect_regenerated_libero_points.py \
  --dataset_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/ObjectFlow/datasets/libero_object_no_noops \
  --output_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/ObjectFlow/datasets/libero_object_no_noops_inspect \
  --num_visualize 12 \
  --sample_points 512 \
  --error_tol 1e-6
```

### Output

- Summary CSV: `<output_dir>/summary_metrics.csv`
- Plot images: `<output_dir>/plots/*.png`
- Console summary with global `max_abs_error` and `rmse`

## 3. Visualize Full Demo Point Flow as Video

### Purpose

Randomly choose one complete demo from regenerated dataset and render scene point cloud evolution frame-by-frame.

- Point color: displacement magnitude `|disp|`
- Optional red arrows: sparse flow vectors

### Command

```bash
python experiments/robot/libero/generate_dataset/visualize_libero_pointflow_video.py \
  --dataset_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/ObjectFlow/datasets/libero_object_no_noops \
  --output_video /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/ObjectFlow/datasets/libero_object_no_noops/random_demo_pointflow.mp4 \
  --seed 7 \
  --max_points 1024 \
  --arrow_stride 24 \
  --fps 12 \
  --elev 22 \
  --azim -55
```

### Key Arguments

- `--seed`: controls which random demo is selected
- `--max_points`: max rendered points per frame
- `--arrow_stride`: draw one arrow every N points (`<=0` means no arrows)
- `--fps`: output video FPS
- `--elev`, `--azim`: 3D camera angles

### Output

- Video file at `--output_video`
- If mp4 encoding fails, script falls back to GIF and raises a message

## Data Format Reference (Regenerated HDF5)

Typical path for one episode:

- `/data/demo_x/obs/pointcloud_abs`: `(T, Np, 3)`, `float32`
- `/data/demo_x/obs/pointcloud_disp`: `(T-1, Np, 3)`, `float32`
- `/data/demo_x/obs/point_track_geom_ids`: `(Np,)`, `int32`
- `/data/demo_x/obs/point_track_face_indices`: `(Np,)`, `int32`
- `/data/demo_x/obs/point_track_face_vertex_indices`: `(Np, 3)`, `int32`
- `/data/demo_x/obs/point_track_barycentric`: `(Np, 3)`, `float32`

## Maintenance Rule

When any dataset script in this folder is added/removed/changed (arguments, behavior, outputs), update this `README.md` in the same commit.
