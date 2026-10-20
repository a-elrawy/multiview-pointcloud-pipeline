import numpy as np
from typing import List, Tuple
import logging

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logger = logging.getLogger(__name__)


def generate_synthetic_scene(
    num_views: int = 4,
    points_per_view: int = 5000,
    noise_std: float = 0.003,
    outlier_ratio: float = 0.04,
    random_seed: int = 42,
) -> Tuple[List["o3d.geometry.PointCloud"], List[np.ndarray]]:
    """Generate synthetic multi-view point cloud scans of a 3D target.

    Simulates realistic multi-angle sensor capture with:
    - Directional visibility (points visible to camera)
    - Additive sensor noise
    - Outlier scatter
    - Rigid SE(3) spatial transformations between scans

    Returns:
        scans: List of misaligned open3d.geometry.PointCloud objects.
        ground_truth_poses: List of 4x4 transformation matrices.
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for synthetic point cloud generation.")

    rng = np.random.default_rng(random_seed)

    # 1. Base surface: Torus + Sphere combination
    u = rng.uniform(0, 2 * np.pi, 25000)
    v = rng.uniform(0, 2 * np.pi, 25000)
    R, r = 1.0, 0.4
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    torus_pts = np.stack([x, y, z], axis=1)

    # Add an inner sphere
    phi = rng.uniform(0, np.pi, 8000)
    theta = rng.uniform(0, 2 * np.pi, 8000)
    rs = 0.35
    sx = rs * np.sin(phi) * np.cos(theta)
    sy = rs * np.sin(phi) * np.sin(theta)
    sz = rs * np.cos(phi)
    sphere_pts = np.stack([sx, sy, sz], axis=1)

    full_surface = np.vstack([torus_pts, sphere_pts])

    # 2. Setup camera viewpoints around the origin
    scans = []
    ground_truth_poses = []

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        cam_pos = np.array([2.5 * np.cos(angle), 2.5 * np.sin(angle), 1.0])
        look_at = np.array([0.0, 0.0, 0.0])

        # Forward vector
        fwd = look_at - cam_pos
        fwd /= np.linalg.norm(fwd)

        # Visibility filter: dot product with surface normal (approximate via position)
        vec_to_cam = cam_pos - full_surface
        dists = np.linalg.norm(vec_to_cam, axis=1, keepdims=True)
        dir_to_cam = vec_to_cam / dists

        # Ray-angle visibility proxy
        normals = full_surface.copy()
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        visible_mask = np.sum(dir_to_cam * normals, axis=1) > 0.15

        sub_pts = full_surface[visible_mask]
        if len(sub_pts) > points_per_view:
            idx = rng.choice(len(sub_pts), points_per_view, replace=False)
            sub_pts = sub_pts[idx]

        # Add Gaussian sensor noise
        sub_pts += rng.normal(0, noise_std, sub_pts.shape)

        # Add stray outlier points
        n_outliers = int(len(sub_pts) * outlier_ratio)
        if n_outliers > 0:
            box_min, box_max = full_surface.min(0) * 1.5, full_surface.max(0) * 1.5
            outliers = rng.uniform(box_min, box_max, (n_outliers, 3))
            sub_pts = np.vstack([sub_pts, outliers])

        # Create realistic initial misalignment (perturbation SE(3))
        # Small perturbation so ICP/RANSAC can register cleanly
        rot_angle = rng.normal(0, np.radians(8.0))
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)

        # Rodrigues rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R_mat = np.eye(3) + np.sin(rot_angle) * K + (1 - np.cos(rot_angle)) * (K @ K)
        t_vec = rng.normal(0, 0.05, 3)

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t_vec

        # Apply perturbation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sub_pts)
        pcd.transform(T)

        # Set distinctive synthetic RGB colors per view for visualization
        colors = np.zeros_like(sub_pts)
        colors[:, i % 3] = 0.8
        colors[:, (i + 1) % 3] = 0.2 + 0.1 * i
        pcd.colors = o3d.utility.Vector3dVector(colors)

        scans.append(pcd)
        ground_truth_poses.append(np.linalg.inv(T))

    logger.info("Generated %d synthetic scans (%d points each).", num_views, points_per_view)
    return scans, ground_truth_poses
