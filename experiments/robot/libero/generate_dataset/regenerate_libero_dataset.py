"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR>

    Example (LIBERO-Spatial):

        export PYTHONPATH=$PYTHONPATH:/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/LIBERO 
        export PYTHONPATH=$PYTHONPATH:/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/diffusion_policy

        python experiments/robot/libero/regenerate_libero_dataset.py \
            --libero_task_suite libero_object \
            --libero_raw_data_dir /inspire/hdd/project/wuliqifa/public/dataset/libero/datasets/libero_object \
            --libero_target_dir /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/ObjectFlow/datasets/libero_object_no_noops

"""

import argparse
import json
import os

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from  libero.libero import benchmark

try:
    import mujoco

    MJ_GEOM_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
    MJ_GEOM_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    MJ_GEOM_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    MJ_GEOM_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    MJ_GEOM_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
    MJ_GEOM_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
except Exception:
    MJ_GEOM_SPHERE = 2
    MJ_GEOM_CAPSULE = 3
    MJ_GEOM_ELLIPSOID = 4
    MJ_GEOM_CYLINDER = 5
    MJ_GEOM_BOX = 6
    # Fallback value used by MuJoCo enums for mesh geom type.
    MJ_GEOM_MESH = 7

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)


IMAGE_RESOLUTION = 256
ROBOT_NAME_TOKENS = (
    "robot",
    "panda",
    "franka",
    "ur",
    "kinova",
    "sawyer",
    "gripper",
    "wrist",
    "arm",
)
PHASE_ID_TO_NAME = {
    0: "idle",
    1: "robot_move",
    2: "grasp_or_contact",
    3: "co_move",
    4: "object_switch",
    5: "object_move",
}


def _get_robot_center_world(env, obs):
    """Get a robot-centered world coordinate for mesh cropping."""
    if "robot0_eef_pos" in obs:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)

    sim = env.sim
    if hasattr(env, "robots") and len(env.robots) > 0:
        root_body = getattr(env.robots[0].robot_model, "root_body", None)
        if root_body is not None:
            try:
                body_id = sim.model.body_name2id(root_body)
                return np.asarray(sim.data.body_xpos[body_id], dtype=np.float32)
            except Exception:
                pass

    # Last-resort fallback: use first body world position.
    return np.asarray(sim.data.body_xpos[0], dtype=np.float32)


def _collect_mesh_triangles_in_cube(sim, cube_center, cube_size):
    """Collect mesh triangles within a robot-centered 3D cube.

    Returns a dict of arrays:
        geom_ids: (M,)
        face_indices: (M,) local face index inside each mesh asset
        face_vertex_indices: (M, 3) local vertex indices in mesh asset
        tri_local: (M, 3, 3) vertices in geom local coordinates (after mesh scale)
        area: (M,) world-space triangle area
        is_robot_face: (M,) whether face belongs to robot mesh geom
    """
    model, data = sim.model, sim.data

    geom_type = np.asarray(model.geom_type)
    geom_dataid = np.asarray(model.geom_dataid)
    geom_xpos = np.asarray(data.geom_xpos)
    geom_xmat = np.asarray(data.geom_xmat).reshape(-1, 3, 3)

    mesh_vert = np.asarray(model.mesh_vert)
    mesh_face = np.asarray(model.mesh_face, dtype=np.int32)
    mesh_vertadr = np.asarray(model.mesh_vertadr, dtype=np.int32)
    mesh_vertnum = np.asarray(model.mesh_vertnum, dtype=np.int32)
    mesh_faceadr = np.asarray(model.mesh_faceadr, dtype=np.int32)
    mesh_facenum = np.asarray(model.mesh_facenum, dtype=np.int32)
    mesh_scale = np.asarray(model.mesh_scale) if hasattr(model, "mesh_scale") else None

    half_extent = cube_size / 2.0

    geom_ids_all = []
    face_idx_all = []
    face_vidx_all = []
    tri_local_all = []
    area_all = []
    is_robot_all = []

    for geom_id in range(model.ngeom):
        if int(geom_type[geom_id]) != MJ_GEOM_MESH:
            continue

        mesh_id = int(geom_dataid[geom_id])
        if mesh_id < 0:
            continue

        vert_start = mesh_vertadr[mesh_id]
        vert_num = mesh_vertnum[mesh_id]
        face_start = mesh_faceadr[mesh_id]
        face_num = mesh_facenum[mesh_id]
        if vert_num == 0 or face_num == 0:
            continue

        verts_local = mesh_vert[vert_start : vert_start + vert_num].copy()
        if mesh_scale is not None and len(mesh_scale) > mesh_id:
            verts_local = verts_local * mesh_scale[mesh_id]

        faces_local = mesh_face[face_start : face_start + face_num].copy()
        tri_local = verts_local[faces_local]  # (F, 3, 3)

        # Transform local -> world by current geom pose.
        rotation = geom_xmat[geom_id]
        translation = geom_xpos[geom_id]
        tri_world = tri_local @ rotation.T + translation

        tri_centers = tri_world.mean(axis=1)
        in_cube = np.all(np.abs(tri_centers - cube_center[None, :]) <= half_extent, axis=1)
        if not np.any(in_cube):
            continue

        tri_world_kept = tri_world[in_cube]
        edge_1 = tri_world_kept[:, 1] - tri_world_kept[:, 0]
        edge_2 = tri_world_kept[:, 2] - tri_world_kept[:, 0]
        tri_area = 0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1)

        kept_face_indices = np.nonzero(in_cube)[0].astype(np.int32)
        kept_face_vidx = faces_local[in_cube].astype(np.int32)
        geom_name = model.geom_id2name(geom_id)
        body_name = model.body_id2name(int(model.geom_bodyid[geom_id]))
        text = f"{geom_name or ''} {body_name or ''}".lower()
        is_robot_geom = any(token in text for token in ROBOT_NAME_TOKENS)

        geom_ids_all.append(np.full((kept_face_indices.shape[0],), geom_id, dtype=np.int32))
        face_idx_all.append(kept_face_indices)
        face_vidx_all.append(kept_face_vidx)
        tri_local_all.append(tri_local[in_cube].astype(np.float32))
        area_all.append(tri_area.astype(np.float64))
        is_robot_all.append(np.full((kept_face_indices.shape[0],), is_robot_geom, dtype=bool))

    if len(area_all) == 0:
        raise RuntimeError(
            "No mesh triangles found in the robot-centered crop cube. "
            "Try increasing `--point_cube_size` or verify mesh assets in environment."
        )

    return {
        "geom_ids": np.concatenate(geom_ids_all, axis=0),
        "face_indices": np.concatenate(face_idx_all, axis=0),
        "face_vertex_indices": np.concatenate(face_vidx_all, axis=0),
        "tri_local": np.concatenate(tri_local_all, axis=0),
        "area": np.concatenate(area_all, axis=0),
        "is_robot_face": np.concatenate(is_robot_all, axis=0),
    }


def _triangulate_primitive_geom_local(geom_type, geom_size, angular_bins=24, lat_bins=12):
    """Generate local-space surface triangles for primitive MuJoCo geoms."""
    size = np.asarray(geom_size, dtype=np.float32)
    pi = np.pi

    if geom_type == MJ_GEOM_BOX:
        hx, hy, hz = float(size[0]), float(size[1]), float(size[2])
        vertices = np.array(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [hx, hy, -hz],
                [-hx, hy, -hz],
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [hx, hy, hz],
                [-hx, hy, hz],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 6, 5], [4, 7, 6],
                [0, 4, 5], [0, 5, 1],
                [1, 5, 6], [1, 6, 2],
                [2, 6, 7], [2, 7, 3],
                [3, 7, 4], [3, 4, 0],
            ],
            dtype=np.int32,
        )
        return vertices[faces]

    if geom_type == MJ_GEOM_CYLINDER:
        radius = float(size[0])
        half_len = float(size[1])
        theta = np.linspace(0.0, 2.0 * pi, angular_bins, endpoint=False)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        tris = []
        for i in range(angular_bins):
            j = (i + 1) % angular_bins
            p0 = np.array([x[i], y[i], -half_len], dtype=np.float32)
            p1 = np.array([x[j], y[j], -half_len], dtype=np.float32)
            p2 = np.array([x[j], y[j], half_len], dtype=np.float32)
            p3 = np.array([x[i], y[i], half_len], dtype=np.float32)
            tris.append(np.stack([p0, p1, p2], axis=0))
            tris.append(np.stack([p0, p2, p3], axis=0))

            c_bot = np.array([0.0, 0.0, -half_len], dtype=np.float32)
            c_top = np.array([0.0, 0.0, half_len], dtype=np.float32)
            tris.append(np.stack([c_bot, p1, p0], axis=0))
            tris.append(np.stack([c_top, p3, p2], axis=0))

        return np.stack(tris, axis=0).astype(np.float32)

    if geom_type in (MJ_GEOM_SPHERE, MJ_GEOM_ELLIPSOID):
        if geom_type == MJ_GEOM_SPHERE:
            radii = np.array([size[0], size[0], size[0]], dtype=np.float32)
        else:
            radii = np.array([size[0], size[1], size[2]], dtype=np.float32)

        lon = np.linspace(0.0, 2.0 * pi, angular_bins + 1)
        lat = np.linspace(-0.5 * pi, 0.5 * pi, lat_bins + 1)
        tris = []
        for il in range(lat_bins):
            lat0, lat1 = lat[il], lat[il + 1]
            for io in range(angular_bins):
                lon0, lon1 = lon[io], lon[io + 1]
                p00 = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)], dtype=np.float32)
                p01 = np.array([np.cos(lat0) * np.cos(lon1), np.cos(lat0) * np.sin(lon1), np.sin(lat0)], dtype=np.float32)
                p10 = np.array([np.cos(lat1) * np.cos(lon0), np.cos(lat1) * np.sin(lon0), np.sin(lat1)], dtype=np.float32)
                p11 = np.array([np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)], dtype=np.float32)

                p00 = p00 * radii
                p01 = p01 * radii
                p10 = p10 * radii
                p11 = p11 * radii
                tris.append(np.stack([p00, p01, p11], axis=0))
                tris.append(np.stack([p00, p11, p10], axis=0))
        return np.stack(tris, axis=0).astype(np.float32)

    if geom_type == MJ_GEOM_CAPSULE:
        radius = float(size[0])
        half_len = float(size[1])
        theta = np.linspace(0.0, 2.0 * pi, angular_bins + 1)
        lat = np.linspace(-0.5 * pi, 0.5 * pi, lat_bins + 1)
        tris = []

        # Cylinder belt.
        for i in range(angular_bins):
            th0, th1 = theta[i], theta[i + 1]
            p0 = np.array([radius * np.cos(th0), radius * np.sin(th0), -half_len], dtype=np.float32)
            p1 = np.array([radius * np.cos(th1), radius * np.sin(th1), -half_len], dtype=np.float32)
            p2 = np.array([radius * np.cos(th1), radius * np.sin(th1), half_len], dtype=np.float32)
            p3 = np.array([radius * np.cos(th0), radius * np.sin(th0), half_len], dtype=np.float32)
            tris.append(np.stack([p0, p1, p2], axis=0))
            tris.append(np.stack([p0, p2, p3], axis=0))

        # Hemispheres.
        for il in range(lat_bins):
            lat0, lat1 = lat[il], lat[il + 1]
            for io in range(angular_bins):
                lon0, lon1 = theta[io], theta[io + 1]
                for sign in (-1.0, 1.0):
                    if sign < 0 and (lat0 > 0 or lat1 > 0):
                        continue
                    if sign > 0 and (lat0 < 0 or lat1 < 0):
                        continue
                    z_shift = sign * half_len
                    p00 = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)], dtype=np.float32)
                    p01 = np.array([np.cos(lat0) * np.cos(lon1), np.cos(lat0) * np.sin(lon1), np.sin(lat0)], dtype=np.float32)
                    p10 = np.array([np.cos(lat1) * np.cos(lon0), np.cos(lat1) * np.sin(lon0), np.sin(lat1)], dtype=np.float32)
                    p11 = np.array([np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)], dtype=np.float32)
                    for p in (p00, p01, p10, p11):
                        p *= radius
                        p[2] += z_shift
                    tris.append(np.stack([p00, p01, p11], axis=0))
                    tris.append(np.stack([p00, p11, p10], axis=0))

        return np.stack(tris, axis=0).astype(np.float32)

    return None


def _collect_surface_triangles_in_cube(sim, cube_center, cube_size):
    """Collect surface triangles for mesh + primitive geoms within a robot-centered 3D cube."""
    model, data = sim.model, sim.data

    geom_type = np.asarray(model.geom_type)
    geom_dataid = np.asarray(model.geom_dataid)
    geom_size = np.asarray(model.geom_size)
    geom_xpos = np.asarray(data.geom_xpos)
    geom_xmat = np.asarray(data.geom_xmat).reshape(-1, 3, 3)

    mesh_vert = np.asarray(model.mesh_vert)
    mesh_face = np.asarray(model.mesh_face, dtype=np.int32)
    mesh_vertadr = np.asarray(model.mesh_vertadr, dtype=np.int32)
    mesh_vertnum = np.asarray(model.mesh_vertnum, dtype=np.int32)
    mesh_faceadr = np.asarray(model.mesh_faceadr, dtype=np.int32)
    mesh_facenum = np.asarray(model.mesh_facenum, dtype=np.int32)
    mesh_scale = np.asarray(model.mesh_scale) if hasattr(model, "mesh_scale") else None

    supported_primitive_types = {
        MJ_GEOM_BOX,
        MJ_GEOM_SPHERE,
        MJ_GEOM_ELLIPSOID,
        MJ_GEOM_CYLINDER,
        MJ_GEOM_CAPSULE,
    }

    half_extent = cube_size / 2.0

    geom_ids_all = []
    face_idx_all = []
    face_vidx_all = []
    tri_local_all = []
    area_all = []
    is_robot_all = []

    for geom_id in range(model.ngeom):
        gtype = int(geom_type[geom_id])
        tri_local = None
        face_vidx = None

        if gtype == MJ_GEOM_MESH:
            mesh_id = int(geom_dataid[geom_id])
            if mesh_id < 0:
                continue

            vert_start = mesh_vertadr[mesh_id]
            vert_num = mesh_vertnum[mesh_id]
            face_start = mesh_faceadr[mesh_id]
            face_num = mesh_facenum[mesh_id]
            if vert_num == 0 or face_num == 0:
                continue

            verts_local = mesh_vert[vert_start : vert_start + vert_num].copy()
            if mesh_scale is not None and len(mesh_scale) > mesh_id:
                verts_local = verts_local * mesh_scale[mesh_id]

            faces_local = mesh_face[face_start : face_start + face_num].copy()
            tri_local = verts_local[faces_local].astype(np.float32)
            face_vidx = faces_local.astype(np.int32)
        elif gtype in supported_primitive_types:
            tri_local = _triangulate_primitive_geom_local(gtype, geom_size[geom_id])
            if tri_local is None or tri_local.shape[0] == 0:
                continue
            face_vidx = np.full((tri_local.shape[0], 3), -1, dtype=np.int32)
        else:
            continue

        rotation = geom_xmat[geom_id]
        translation = geom_xpos[geom_id]
        tri_world = tri_local @ rotation.T + translation

        tri_centers = tri_world.mean(axis=1)
        in_cube = np.all(np.abs(tri_centers - cube_center[None, :]) <= half_extent, axis=1)
        if not np.any(in_cube):
            continue

        tri_world_kept = tri_world[in_cube]
        edge_1 = tri_world_kept[:, 1] - tri_world_kept[:, 0]
        edge_2 = tri_world_kept[:, 2] - tri_world_kept[:, 0]
        tri_area = 0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1)

        kept_face_indices = np.nonzero(in_cube)[0].astype(np.int32)
        kept_face_vidx = face_vidx[in_cube].astype(np.int32)
        geom_name = model.geom_id2name(geom_id)
        body_name = model.body_id2name(int(model.geom_bodyid[geom_id]))
        text = f"{geom_name or ''} {body_name or ''}".lower()
        is_robot_geom = any(token in text for token in ROBOT_NAME_TOKENS)

        geom_ids_all.append(np.full((kept_face_indices.shape[0],), geom_id, dtype=np.int32))
        face_idx_all.append(kept_face_indices)
        face_vidx_all.append(kept_face_vidx)
        tri_local_all.append(tri_local[in_cube].astype(np.float32))
        area_all.append(tri_area.astype(np.float64))
        is_robot_all.append(np.full((kept_face_indices.shape[0],), is_robot_geom, dtype=bool))

    if len(area_all) == 0:
        raise RuntimeError(
            "No surface triangles found in the robot-centered crop cube. "
            "Try increasing `--point_cube_size` or verify geom assets in environment."
        )

    return {
        "geom_ids": np.concatenate(geom_ids_all, axis=0),
        "face_indices": np.concatenate(face_idx_all, axis=0),
        "face_vertex_indices": np.concatenate(face_vidx_all, axis=0),
        "tri_local": np.concatenate(tri_local_all, axis=0),
        "area": np.concatenate(area_all, axis=0),
        "is_robot_face": np.concatenate(is_robot_all, axis=0),
    }


def _sample_point_tracks_from_mesh(
    sim,
    env,
    obs,
    n_points,
    cube_size,
    rng,
    robot_point_weight,
    min_non_robot_ratio,
):
    """Initialize point tracks by mesh-face sampling + barycentric coordinates.

    This follows Pri4R-style initialization:
      1) Crop mesh triangles inside robot-centered 3D cube.
      2) Uniformly sample surface points over mesh faces (area-weighted).
      3) Store face indices + barycentric coordinates for identity-consistent tracking.
    """
    center = _get_robot_center_world(env, obs)
    mesh_data = _collect_surface_triangles_in_cube(sim, center, cube_size)

    areas = mesh_data["area"]
    is_robot_face = mesh_data["is_robot_face"]
    if np.sum(areas) <= 0:
        raise RuntimeError("All candidate mesh face areas are zero; cannot sample track points.")

    # Reduce robot over-sampling by down-weighting robot mesh faces.
    weighted_areas = areas.copy()
    if robot_point_weight <= 0:
        raise ValueError(f"robot_point_weight must be > 0, got {robot_point_weight}")
    weighted_areas[is_robot_face] *= robot_point_weight

    if np.sum(weighted_areas) <= 0:
        raise RuntimeError("All weighted mesh areas are zero after robot weighting.")

    # Optionally enforce a minimum fraction of non-robot points.
    has_robot = np.any(is_robot_face)
    has_non_robot = np.any(~is_robot_face)
    enforce_non_robot = min_non_robot_ratio > 0 and has_non_robot
    if enforce_non_robot:
        target_non_robot = int(np.ceil(n_points * min_non_robot_ratio))
        target_non_robot = min(max(target_non_robot, 0), n_points)
        target_robot = n_points - target_non_robot

        non_robot_ids = np.nonzero(~is_robot_face)[0]
        non_robot_w = weighted_areas[non_robot_ids]
        non_robot_prob = non_robot_w / np.sum(non_robot_w)
        sampled_non_robot = rng.choice(non_robot_ids, size=target_non_robot, replace=True, p=non_robot_prob)

        sampled_robot = np.array([], dtype=np.int64)
        if target_robot > 0:
            if has_robot:
                robot_ids = np.nonzero(is_robot_face)[0]
                robot_w = weighted_areas[robot_ids]
                robot_prob = robot_w / np.sum(robot_w)
                sampled_robot = rng.choice(robot_ids, size=target_robot, replace=True, p=robot_prob)
            else:
                sampled_robot = rng.choice(non_robot_ids, size=target_robot, replace=True, p=non_robot_prob)

        sampled_face_ids = np.concatenate([sampled_non_robot, sampled_robot], axis=0)
        rng.shuffle(sampled_face_ids)
    else:
        face_prob = weighted_areas / np.sum(weighted_areas)
        sampled_face_ids = rng.choice(np.arange(len(face_prob)), size=n_points, replace=True, p=face_prob)

    # Uniform point sampling on triangle surface via barycentric coordinates.
    # See Turk 1990: use sqrt trick for uniform area sampling.
    rand_1 = rng.random(n_points)
    rand_2 = rng.random(n_points)
    sqrt_rand_1 = np.sqrt(rand_1)
    barycentric = np.stack(
        [
            1.0 - sqrt_rand_1,
            sqrt_rand_1 * (1.0 - rand_2),
            sqrt_rand_1 * rand_2,
        ],
        axis=1,
    ).astype(np.float32)  # (Np, 3)

    sampled_tri_local = mesh_data["tri_local"][sampled_face_ids]  # (Np, 3, 3)
    sampled_local_points = np.sum(sampled_tri_local * barycentric[:, :, None], axis=1).astype(np.float32)  # (Np, 3)

    return {
        "geom_ids": mesh_data["geom_ids"][sampled_face_ids].astype(np.int32),
        "face_indices": mesh_data["face_indices"][sampled_face_ids].astype(np.int32),
        "face_vertex_indices": mesh_data["face_vertex_indices"][sampled_face_ids].astype(np.int32),
        "barycentric": barycentric,
        "local_points": sampled_local_points,
        "is_robot_point": mesh_data["is_robot_face"][sampled_face_ids].astype(np.uint8),
    }


def _track_points_world(sim, point_track_info):
    """Track identity-consistent world points using stored barycentric local points."""
    geom_ids = point_track_info["geom_ids"]  # (Np,)
    local_points = point_track_info["local_points"]  # (Np, 3)

    geom_xpos = np.asarray(sim.data.geom_xpos)  # (G, 3)
    geom_xmat = np.asarray(sim.data.geom_xmat).reshape(-1, 3, 3)  # (G, 3, 3)

    world_points = np.zeros_like(local_points, dtype=np.float32)
    for geom_id in np.unique(geom_ids):
        mask = geom_ids == geom_id
        rotation = geom_xmat[geom_id]
        translation = geom_xpos[geom_id]
        world_points[mask] = (local_points[mask] @ rotation.T + translation).astype(np.float32)

    return world_points


def _assign_object_groups(first_active_t, is_robot_point, time_gap):
    """Cluster object points by first activation time to approximate manipulated object groups."""
    n_points = first_active_t.shape[0]
    group_ids = np.full((n_points,), -1, dtype=np.int32)

    valid = np.where((~is_robot_point) & (first_active_t >= 0))[0]
    if valid.shape[0] == 0:
        return group_ids

    ordered = valid[np.argsort(first_active_t[valid])]
    curr_group = 0
    prev_t = int(first_active_t[ordered[0]])
    group_ids[ordered[0]] = curr_group

    for idx in ordered[1:]:
        t = int(first_active_t[idx])
        if t - prev_t > time_gap:
            curr_group += 1
        group_ids[idx] = curr_group
        prev_t = t

    return group_ids


def _extract_pointflow_temporal_info(pointcloud_disp_np, is_robot_point, args):
    """Extract point-wise motion states and coarse operation phases from temporal point flow."""
    n_steps = pointcloud_disp_np.shape[0]
    n_points = is_robot_point.shape[0]

    speed = np.linalg.norm(pointcloud_disp_np, axis=-1).astype(np.float32)  # (T-1, Np)
    is_moving = (speed > args.point_motion_threshold).astype(np.uint8)  # (T-1, Np)

    first_active_t = np.full((n_points,), -1, dtype=np.int32)
    if n_steps > 0:
        move_bool = is_moving.astype(bool)
        any_moved = move_bool.any(axis=0)
        first_idx = np.argmax(move_bool, axis=0).astype(np.int32)
        first_active_t[any_moved] = first_idx[any_moved]

    robot_mask = is_robot_point.astype(bool)
    object_mask = ~robot_mask
    object_group_id = _assign_object_groups(first_active_t, robot_mask, args.object_group_time_gap)

    phase_label = np.zeros((n_steps,), dtype=np.uint8)
    dominant_object_group = np.full((n_steps,), -1, dtype=np.int32)
    robot_active_ratio = np.zeros((n_steps,), dtype=np.float32)
    object_active_ratio = np.zeros((n_steps,), dtype=np.float32)

    prev_object_active = False
    unique_groups = np.unique(object_group_id[object_group_id >= 0])
    for t in range(n_steps):
        if np.any(robot_mask):
            robot_active_ratio[t] = float(is_moving[t, robot_mask].mean())
        if np.any(object_mask):
            object_active_ratio[t] = float(is_moving[t, object_mask].mean())

        robot_active = robot_active_ratio[t] >= args.robot_active_ratio_threshold
        object_active = object_active_ratio[t] >= args.object_active_ratio_threshold

        best_group = -1
        best_group_ratio = 0.0
        for group_id in unique_groups:
            mask = object_group_id == group_id
            if np.any(mask):
                ratio = float(is_moving[t, mask].mean())
                if ratio > best_group_ratio:
                    best_group_ratio = ratio
                    best_group = int(group_id)
        if best_group_ratio >= args.object_group_active_ratio_threshold:
            dominant_object_group[t] = best_group

        if robot_active and not object_active:
            phase_label[t] = 1
        elif robot_active and object_active:
            if not prev_object_active:
                phase_label[t] = 2
            elif (
                t > 0
                and dominant_object_group[t] >= 0
                and dominant_object_group[t - 1] >= 0
                and dominant_object_group[t] != dominant_object_group[t - 1]
            ):
                phase_label[t] = 4
            else:
                phase_label[t] = 3
        elif (not robot_active) and object_active:
            phase_label[t] = 5
        else:
            phase_label[t] = 0

        prev_object_active = bool(object_active)

    return {
        "point_motion_speed": speed,
        "point_motion_is_moving": is_moving,
        "point_motion_first_active_t": first_active_t,
        "point_motion_object_group_id": object_group_id,
        "phase_label": phase_label,
        "phase_dominant_object_group": dominant_object_group,
        "phase_robot_active_ratio": robot_active_ratio,
        "phase_object_active_ratio": object_active_ratio,
    }


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def _normalize_single_hdf5_name(single_hdf5_name):
    """Normalize debug hdf5 selector to '<task_name>_demo.hdf5' format."""
    if single_hdf5_name is None:
        return None

    name = os.path.basename(single_hdf5_name.strip())
    if name.endswith("_demo.hdf5"):
        return name
    if name.endswith(".hdf5"):
        return f"{name[:-5]}_demo.hdf5"
    return f"{name}_demo.hdf5"


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        user_input = input(f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: ")
        if user_input != 'y':
            exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = f"./experiments/robot/libero/{args.libero_task_suite}_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0
    selected_hdf5_name = _normalize_single_hdf5_name(args.single_hdf5_name)
    matched_task_count = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        raw_hdf5_name = f"{task.name}_demo.hdf5"
        if selected_hdf5_name is not None and raw_hdf5_name != selected_hdf5_name:
            continue

        matched_task_count += 1
        if selected_hdf5_name is not None:
            print(f"[debug] Processing only selected file: {raw_hdf5_name}")

        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, raw_hdf5_name)
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        for i in range(len(orig_data.keys())):
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []
            pointcloud_abs = []

            # Pri4R-style 3D point track initialization on first frame.
            episode_rng = np.random.default_rng(seed=args.point_seed + task_id * 100000 + i)
            point_track_info = _sample_point_tracks_from_mesh(
                sim=env.sim,
                env=env,
                obs=obs,
                n_points=args.point_count,
                cube_size=args.point_cube_size,
                rng=episode_rng,
                robot_point_weight=args.robot_point_weight,
                min_non_robot_ratio=args.min_non_robot_ratio,
            )

            # Replay original demo actions in environment and record observations
            for _, action in enumerate(orig_actions):
                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                # Record identity-consistent point cloud in world coordinates.
                curr_points_world = _track_points_world(env.sim, point_track_info)  # (Np, 3)
                pointcloud_abs.append(curr_points_world)

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)
                assert len(pointcloud_abs) == len(actions)

                pointcloud_abs_np = np.stack(pointcloud_abs, axis=0).astype(np.float32)  # (T, Np, 3)
                if pointcloud_abs_np.shape[0] > 1:
                    pointcloud_disp_np = (pointcloud_abs_np[1:] - pointcloud_abs_np[:-1]).astype(np.float32)  # (T-1, Np, 3)
                else:
                    pointcloud_disp_np = np.zeros((0, args.point_count, 3), dtype=np.float32)
                temporal_info = _extract_pointflow_temporal_info(
                    pointcloud_disp_np=pointcloud_disp_np,
                    is_robot_point=point_track_info["is_robot_point"].astype(bool),
                    args=args,
                )

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                obs_grp.create_dataset("pointcloud_abs", data=pointcloud_abs_np, dtype=np.float32)
                obs_grp.create_dataset("pointcloud_disp", data=pointcloud_disp_np, dtype=np.float32)
                obs_grp.create_dataset("point_track_geom_ids", data=point_track_info["geom_ids"], dtype=np.int32)
                obs_grp.create_dataset("point_track_face_indices", data=point_track_info["face_indices"], dtype=np.int32)
                obs_grp.create_dataset("point_track_face_vertex_indices", data=point_track_info["face_vertex_indices"], dtype=np.int32)
                obs_grp.create_dataset("point_track_barycentric", data=point_track_info["barycentric"], dtype=np.float32)
                obs_grp.create_dataset("point_track_is_robot", data=point_track_info["is_robot_point"], dtype=np.uint8)
                obs_grp.create_dataset("point_track_semantic_id", data=point_track_info["is_robot_point"], dtype=np.uint8)
                obs_grp.create_dataset("point_motion_speed", data=temporal_info["point_motion_speed"], dtype=np.float32)
                obs_grp.create_dataset("point_motion_is_moving", data=temporal_info["point_motion_is_moving"], dtype=np.uint8)
                obs_grp.create_dataset("point_motion_first_active_t", data=temporal_info["point_motion_first_active_t"], dtype=np.int32)
                obs_grp.create_dataset("point_motion_object_group_id", data=temporal_info["point_motion_object_group_id"], dtype=np.int32)
                obs_grp.create_dataset("phase_label", data=temporal_info["phase_label"], dtype=np.uint8)
                obs_grp.create_dataset(
                    "phase_dominant_object_group",
                    data=temporal_info["phase_dominant_object_group"],
                    dtype=np.int32,
                )
                obs_grp.create_dataset("phase_robot_active_ratio", data=temporal_info["phase_robot_active_ratio"], dtype=np.float32)
                obs_grp.create_dataset("phase_object_active_ratio", data=temporal_info["phase_object_active_ratio"], dtype=np.float32)
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)

                num_success += 1

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")
            if done and temporal_info["phase_label"].shape[0] > 0:
                phase_counts = {
                    PHASE_ID_TO_NAME[int(pid)]: int((temporal_info["phase_label"] == pid).sum())
                    for pid in np.unique(temporal_info["phase_label"])
                }
                print(f"  Phase counts: {phase_counts}")

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")
    if selected_hdf5_name is not None and matched_task_count == 0:
        raise RuntimeError(
            f"No task matched --single_hdf5_name='{args.single_hdf5_name}' in suite '{args.libero_task_suite}'."
        )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--libero_target_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", required=True)
    parser.add_argument("--point_count", type=int, default=1024,
                        help="Number of tracked mesh-surface points per episode (Pri4R uses 1024).")
    parser.add_argument("--point_cube_size", type=float, default=1.2,
                        help="Robot-centered cube size (meters) used for mesh cropping at first frame.")
    parser.add_argument("--point_seed", type=int, default=7,
                        help="Random seed for mesh-face surface sampling.")
    parser.add_argument(
        "--robot_point_weight",
        type=float,
        default=0.2,
        help="Area weight multiplier for robot mesh faces (<1 reduces robot-point density).",
    )
    parser.add_argument(
        "--min_non_robot_ratio",
        type=float,
        default=0.7,
        help="Minimum fraction of sampled points from non-robot faces (range [0, 1]).",
    )
    parser.add_argument(
        "--single_hdf5_name",
        type=str,
        default=None,
        help=(
            "Debug mode: process only one raw hdf5 file. Accepts task name, task_demo, or task_demo.hdf5; "
            "for example 'KITCHEN_SCENE1_put_the_black_bowl_on_the_plate'."
        ),
    )
    parser.add_argument(
        "--point_motion_threshold",
        type=float,
        default=2e-3,
        help="Motion threshold (meters/frame) for static/moving point state split.",
    )
    parser.add_argument(
        "--robot_active_ratio_threshold",
        type=float,
        default=0.02,
        help="Robot-active threshold on moving-point ratio per timestep.",
    )
    parser.add_argument(
        "--object_active_ratio_threshold",
        type=float,
        default=0.01,
        help="Object-active threshold on moving-point ratio per timestep.",
    )
    parser.add_argument(
        "--object_group_time_gap",
        type=int,
        default=8,
        help="Frame-gap threshold for splitting object groups by first activation time.",
    )
    parser.add_argument(
        "--object_group_active_ratio_threshold",
        type=float,
        default=0.02,
        help="Minimum moving ratio for assigning dominant active object group at a timestep.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.min_non_robot_ratio <= 1.0):
        raise ValueError(f"--min_non_robot_ratio must be in [0, 1], got {args.min_non_robot_ratio}")
    if not (0.0 <= args.robot_active_ratio_threshold <= 1.0):
        raise ValueError(
            f"--robot_active_ratio_threshold must be in [0, 1], got {args.robot_active_ratio_threshold}"
        )
    if not (0.0 <= args.object_active_ratio_threshold <= 1.0):
        raise ValueError(
            f"--object_active_ratio_threshold must be in [0, 1], got {args.object_active_ratio_threshold}"
        )
    if not (0.0 <= args.object_group_active_ratio_threshold <= 1.0):
        raise ValueError(
            "--object_group_active_ratio_threshold must be in [0, 1], "
            f"got {args.object_group_active_ratio_threshold}"
        )
    if args.point_motion_threshold < 0:
        raise ValueError(f"--point_motion_threshold must be >= 0, got {args.point_motion_threshold}")
    if args.object_group_time_gap < 0:
        raise ValueError(f"--object_group_time_gap must be >= 0, got {args.object_group_time_gap}")

    # Start data regeneration
    main(args)
