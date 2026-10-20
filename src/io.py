import os
import logging
from typing import List, Optional, Union
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logger = logging.getLogger(__name__)


def load_point_cloud(file_path: str):
    """Load a point cloud from disk (.ply, .pcd, .xyz)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if HAS_OPEN3D:
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            raise ValueError(f"Failed to load or empty point cloud: {file_path}")
        return pcd
    else:
        # Fallback parser for basic text/xyz or numpy files
        if file_path.endswith((".npy", ".npz")):
            pts = np.load(file_path)
            return pts
        raise ImportError("open3d is required to load standard 3D formats (.ply, .pcd).")


def save_point_cloud(pcd, file_path: str, write_ascii: bool = False):
    """Save a point cloud to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    if HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(file_path, pcd, write_ascii=write_ascii)
        logger.info("Saved point cloud (%d points) to %s", len(pcd.points), file_path)
    else:
        raise ImportError("open3d is required to write standard point cloud formats.")


def save_mesh(mesh, file_path: str, write_ascii: bool = False):
    """Save a triangle mesh to disk (.ply, .obj)."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    if HAS_OPEN3D and isinstance(mesh, o3d.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh(file_path, mesh, write_ascii=write_ascii)
        logger.info(
            "Saved mesh (%d vertices, %d triangles) to %s",
            len(mesh.vertices),
            len(mesh.triangles),
            file_path,
        )
    else:
        raise ImportError("open3d is required to write mesh formats.")
