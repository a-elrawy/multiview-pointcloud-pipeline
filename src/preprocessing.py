import logging
from typing import Dict, Any, Optional
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logger = logging.getLogger(__name__)


def estimate_normals(
    pcd: "o3d.geometry.PointCloud",
    radius: float = 0.05,
    max_nn: int = 30,
    orient_towards_camera: bool = True,
    camera_location: Optional[np.ndarray] = None,
) -> "o3d.geometry.PointCloud":
    """Estimate surface normals using local covariance analysis (PCA).

    Orient normals coherently either towards a given camera location
    or towards the bounding box center.
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for normal estimation.")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )

    if orient_towards_camera:
        if camera_location is None:
            camera_location = np.array([0.0, 0.0, 0.0])
        pcd.orient_normals_towards_camera_location(camera_location=camera_location)
    else:
        pcd.orient_normals_consistent_tangent_plane(k=15)

    return pcd


def remove_outliers(
    pcd: "o3d.geometry.PointCloud",
    method: str = "sor",
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    radius: float = 0.05,
    nb_points: int = 16,
) -> "o3d.geometry.PointCloud":
    """Remove outlier noise using Statistical Outlier Removal (SOR)

    or Radius Outlier Removal (ROR).
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for outlier removal.")

    initial_pts = len(pcd.points)

    if method.lower() == "sor":
        cleaned_pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
    elif method.lower() == "ror":
        cleaned_pcd, ind = pcd.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")

    removed = initial_pts - len(cleaned_pcd.points)
    logger.debug(
        "Outlier filter [%s]: removed %d / %d points (%.1f%%).",
        method.upper(),
        removed,
        initial_pts,
        (removed / max(initial_pts, 1)) * 100,
    )
    return cleaned_pcd


def preprocess_point_cloud(
    pcd: "o3d.geometry.PointCloud",
    voxel_size: float = 0.02,
    sor_enabled: bool = True,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    ror_enabled: bool = False,
    ror_radius: float = 0.05,
    ror_nb_points: int = 16,
    estimate_normal: bool = True,
    normal_radius: Optional[float] = None,
    normal_max_nn: int = 30,
) -> "o3d.geometry.PointCloud":
    """Full preprocessing pipeline:

    1. Voxel grid downsampling
    2. Statistical and/or Radius outlier removal
    3. Normal estimation with orientation
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for preprocessing.")

    processed = pcd

    # 1. Voxel downsample
    if voxel_size > 0:
        processed = processed.voxel_down_sample(voxel_size=voxel_size)

    # 2. Outlier removal
    if sor_enabled:
        processed = remove_outliers(
            processed,
            method="sor",
            nb_neighbors=sor_nb_neighbors,
            std_ratio=sor_std_ratio,
        )

    if ror_enabled:
        processed = remove_outliers(
            processed,
            method="ror",
            radius=ror_radius,
            nb_points=ror_nb_points,
        )

    # 3. Normal estimation
    if estimate_normal:
        nr = normal_radius if normal_radius is not None else voxel_size * 2.5
        processed = estimate_normals(
            processed,
            radius=nr,
            max_nn=normal_max_nn,
            orient_towards_camera=True,
        )

    return processed
