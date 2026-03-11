# LIBERO Dataset Generation Tools

This folder contains scripts for regenerating LIBERO HDF5 data, validating saved point-flow fields, and visualizing scene point cloud dynamics.

## Files

- `regenerate_libero_dataset.py`: Replay raw LIBERO demos and regenerate filtered HDF5 dataset with point tracks.
- `inspect_regenerated_libero_points.py`: Validate `pointcloud_abs` / `pointcloud_disp` consistency and export diagnostic plots.
- `visualize_libero_pointflow_video.py`: Randomly pick one complete demo and render scene point-flow as a video.
- `visualize_libero_pointflow_on_original_cloud.py`: Render flow directly on saved original `pointcloud_abs` (no re-sampling by default).

## 0. Environment Preparation

Use your `objectflow` environment and make sure LIBERO-related paths are visible:

```bash
export PYTHONPATH=$PYTHONPATH:/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/LIBERO
export PYTHONPATH=$PYTHONPATH:/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/diffusion_policy
export MUJOCO_GL=glx
Xvfb :1 -screen 0 1920x1080x24 &
export DISPLAY=:1


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
- saving Pri4R-style point tracks (`pointcloud_abs`, `pointcloud_disp`, etc.) from both mesh and primitive geoms,
- extracting semantic-temporal point-flow labels (robot vs object, moving-state changes, phase labels).

### Command

```bash
python experiments/robot/libero/generate_dataset/regenerate_libero_dataset.py \
  --libero_task_suite libero_object \
  --libero_raw_data_dir /inspire/hdd/project/wuliqifa/public/dataset/libero/datasets/libero_object \
  --libero_target_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_no_noops \
  --point_count 1024 \
  --point_cube_size 1.2 \
  --point_seed 7 \
  --robot_point_weight 0.2 \
  --min_non_robot_ratio 0.7 \
  --point_motion_threshold 0.002 \
  --robot_active_ratio_threshold 0.02 \
  --object_active_ratio_threshold 0.01 \
  --object_group_active_ratio_threshold 0.02
```

Single-file debug mode (only one task hdf5):

```bash
python experiments/robot/libero/generate_dataset/regenerate_libero_dataset.py \
  --libero_task_suite libero_object \
  --libero_raw_data_dir /inspire/hdd/project/wuliqifa/public/dataset/libero/datasets/libero_object \
  --libero_target_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug \
  --single_hdf5_name pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5
```

### Key Arguments

- `--libero_task_suite`: one of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`
- `--libero_raw_data_dir`: raw LIBERO HDF5 folder
- `--libero_target_dir`: output folder for regenerated HDF5
- `--point_count`: number of sampled tracked points per episode
- `--point_cube_size`: robot-centered crop cube size in meters
- `--point_seed`: random seed for point sampling
- `--robot_point_weight`: area weight multiplier for robot faces (`<1` reduces robot point density)
- `--min_non_robot_ratio`: minimum fraction of sampled points forced to come from non-robot faces
- `--single_hdf5_name`: debug mode, only process one task hdf5 file
- `--point_motion_threshold`: per-point static/moving threshold (meters/frame)
- `--robot_active_ratio_threshold`: phase split threshold on robot moving-point ratio
- `--object_active_ratio_threshold`: phase split threshold on object moving-point ratio
- `--object_group_active_ratio_threshold`: minimum active ratio for dominant object group assignment

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
  --dataset_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_no_noops \
  --output_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_no_noops_inspect \
  --num_visualize 12 \
  --sample_points 512 \
  --error_tol 1e-6
```

Inspect command for single-file debug output:

```bash
python experiments/robot/libero/generate_dataset/inspect_regenerated_libero_points.py \
  --dataset_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug \
  --output_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug_inspect \
  --num_visualize 8 \
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

- Robot points and object points are visualized as separate semantic streams
- Object points are color-coded by moved-object labels (`background`, `move_object_a`, `move_object_b`, ...)
- Optional red arrows: sparse flow vectors
- Optional semantic trails: robot trails vs object-group trails
- Current moving points are highlighted
- Also saves corresponding LIBERO 2D observation video for direct comparison
- Optional side-by-side comparison video (left: 3D point-flow, right: 2D observation)
- A fixed right sidebar legend is rendered per frame to explicitly show semantic/group color mapping

### Command

```bash
python experiments/robot/libero/generate_dataset/visualize_libero_pointflow_video.py \
  --dataset_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_no_noops \
  --output_video /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_no_noops/random_demo_pointflow.mp4 \
  --output_2d_video /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_no_noops/random_demo_agentview.mp4 \
  --save_comparison \
  --output_comparison_video /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_no_noops/random_demo_compare.mp4 \
  --rgb_key agentview_rgb \
  --rotate_rgb_180 \
  --seed 7 \
  --max_points 1024 \
  --arrow_stride 24 \
  --trail_stride 16 \
  --trail_len 25 \
  --trail_alpha 0.35 \
  --trail_width 0.8 \
  --fps 12 \
  --elev 22 \
  --azim -55
```

### Command for single-file debug output:

```bash
python experiments/robot/libero/generate_dataset/visualize_libero_pointflow_video.py \
  --dataset_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug \
  --output_video /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug/random_demo_pointflow.mp4 \
  --output_2d_video /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug/random_demo_agentview.mp4 \
  --save_comparison \
  --output_comparison_video /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug/random_demo_compare.mp4 \
  --rgb_key agentview_rgb \
  --rotate_rgb_180 \
  --seed 7 \
  --max_points 1024 \
  --arrow_stride 24 \
  --trail_stride 16 \
  --trail_len 25 \
  --trail_alpha 0.35 \
  --trail_width 0.8 \
  --fps 12 \
  --elev 22 \
  --azim -55
```

### Key Arguments

- `--seed`: controls which random demo is selected
- `--rgb_key`: 2D stream key, choose from `agentview_rgb` or `eye_in_hand_rgb`
- `--rotate_rgb_180`: rotate 2D RGB frames by 180 degrees before saving (default enabled)
- `--no_rotate_rgb_180`: disable the 180-degree rotation
- `--output_2d_video`: output path of corresponding LIBERO 2D video
- `--save_comparison`: save side-by-side comparison video
- `--output_comparison_video`: output path for side-by-side video
- `--max_points`: max rendered points per frame
- `--arrow_stride`: draw one arrow every N points (`<=0` means no arrows)
- `--trail_stride`: draw one trajectory trail every N points (`<=0` means no trails)
- `--trail_len`: number of recent frames used for each trajectory trail
- `--trail_alpha`: trajectory trail transparency
- `--trail_width`: trajectory trail line width
- `--fps`: output video FPS
- `--elev`, `--azim`: 3D camera angles

In the 3D video title, you will also see:

- `phase=<name>(id)`: coarse operation stage label per frame
- `dom_group=<id>`: dominant active object group at the current frame

In the 3D side legend, you will also see:

- robot color swatch
- moving-point highlight color
- unclustered object color
- per-group color swatches (group id to color mapping)

### Output

- Video file at `--output_video`
- 2D observation video at `--output_2d_video`
- Optional side-by-side video at `--output_comparison_video`
- If mp4 encoding fails, script falls back to GIF and raises a message

## 4. Visualize Flow on Original Saved Point Cloud

### Purpose

Render `pointcloud_disp` directly on the saved original point cloud `pointcloud_abs`.

- By default, no re-sampling is applied (render all saved points)
- This avoids visualization mismatch caused by extra display-time point sub-sampling
- Optional `--max_render_points` is only for speed debugging

### Command

```bash
python experiments/robot/libero/generate_dataset/visualize_libero_pointflow_on_original_cloud.py \
  --dataset_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug \
  --output_video /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug/original_cloud_flow.mp4 \
  --arrow_stride 8 \
  --fps 12
```

Single-file command:

```bash
python experiments/robot/libero/generate_dataset/visualize_libero_pointflow_on_original_cloud.py \
  --hdf5_path /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5 \
  --demo_key demo_0 \
  --output_video /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/datasets/libero_object_debug/original_cloud_demo0.mp4
```

## Data Format Reference (Regenerated HDF5)

Typical path for one episode:

- `/data/demo_x/obs/pointcloud_abs`: `(T, Np, 3)`, `float32`
- `/data/demo_x/obs/pointcloud_disp`: `(T-1, Np, 3)`, `float32`
- `/data/demo_x/obs/point_track_geom_ids`: `(Np,)`, `int32`
- `/data/demo_x/obs/point_track_face_indices`: `(Np,)`, `int32`
- `/data/demo_x/obs/point_track_face_vertex_indices`: `(Np, 3)`, `int32`
- `/data/demo_x/obs/point_track_barycentric`: `(Np, 3)`, `float32`
- `/data/demo_x/obs/point_track_is_robot`: `(Np,)`, `uint8` (`1=robot`, `0=object`)
- `/data/demo_x/obs/point_track_semantic_id`: `(Np,)`, `uint8` (same coding as above)
- `/data/demo_x/obs/point_motion_speed`: `(T-1, Np)`, `float32`
- `/data/demo_x/obs/point_motion_is_moving`: `(T-1, Np)`, `uint8`
- `/data/demo_x/obs/point_motion_first_active_t`: `(Np,)`, `int32` (`-1` means never moved)
- `/data/demo_x/obs/point_motion_object_instance_id`: `(Np,)`, `int32` (`-1` robot, `>=0` object instance id)
- `/data/demo_x/obs/point_motion_object_group_id`: `(Np,)`, `int32` (`-1` robot, `0` background, `>=1` moved objects)
- `/data/demo_x/obs/point_motion_object_label`: `(Np,)`, `S32` (`robot` / `background` / `move_object_*`)
- `/data/demo_x/obs/phase_label`: `(T-1,)`, `uint8`
- `/data/demo_x/obs/phase_dominant_object_group`: `(T-1,)`, `int32`
- `/data/demo_x/obs/phase_robot_active_ratio`: `(T-1,)`, `float32`
- `/data/demo_x/obs/phase_object_active_ratio`: `(T-1,)`, `float32`

Phase label definition:

- `0`: idle
- `1`: robot_move
- `2`: grasp_or_contact
- `3`: co_move
- `4`: object_switch
- `5`: object_move

## Maintenance Rule

When any dataset script in this folder is added/removed/changed (arguments, behavior, outputs), update this `README.md` in the same commit.
