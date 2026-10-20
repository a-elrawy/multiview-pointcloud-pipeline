import logging
from typing import List, Optional
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logger = logging.getLogger(__name__)


def clean_mesh(
    mesh: "o3d.geometry.TriangleMesh",
    remove_unreferenced: bool = True,
    remove_degenerate: bool = True,
    remove_duplicated: bool = True,
    remove_non_manifold: bool = True,
    smooth_iterations: int = 1,
) -> "o3d.geometry.TriangleMesh":
    """Clean mesh topology and remove degeneracies."""
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for mesh processing.")

    if remove_duplicated:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()

    if remove_degenerate:
        mesh.remove_degenerate_triangles()

    if remove_unreferenced:
        mesh.remove_unreferenced_vertices()

    if remove_non_manifold:
        mesh.remove_non_manifold_edges()

    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iterations)

    mesh.compute_vertex_normals()
    return mesh


def reconstruct_surface_poisson(
    pcd: "o3d.geometry.PointCloud",
    depth: int = 9,
    width: float = 0.0,
    scale: float = 1.1,
    linear_fit: bool = False,
    trim_density_quantile: float = 0.05,
) -> "o3d.geometry.TriangleMesh":
    """Screened Poisson Surface Reconstruction with density-based boundary trimming."""
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for Poisson reconstruction.")

    if not pcd.has_normals():
        raise ValueError("Point cloud must contain oriented surface normals for Poisson reconstruction.")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )

    # Trim low-density spurious faces
    if trim_density_quantile > 0.0:
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, trim_density_quantile)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh = clean_mesh(mesh)
    logger.info(
        "Poisson reconstruction finished: %d vertices, %d triangles.",
        len(mesh.vertices),
        len(mesh.triangles),
    )
    return mesh


def reconstruct_surface_bpa(
    pcd: "o3d.geometry.PointCloud",
    radii: Optional[List[float]] = None,
    radii_factors: Optional[List[float]] = None,
) -> "o3d.geometry.TriangleMesh":
    """Ball-Pivoting Algorithm (BPA) surface reconstruction.

    Uses adaptive multi-scale ball radii derived from the average nearest neighbor distance.
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for BPA reconstruction.")

    if not pcd.has_normals():
        raise ValueError("Point cloud must contain surface normals for Ball-Pivoting Algorithm.")

    if radii is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        if radii_factors is None:
            radii_factors = [1.0, 2.0, 4.0]
        radii = [avg_dist * factor for factor in radii_factors]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh = clean_mesh(mesh)
    logger.info(
        "BPA reconstruction finished: %d vertices, %d triangles.",
        len(mesh.vertices),
        len(mesh.triangles),
    )
    return mesh


def reconstruct_surface(
    pcd: "o3d.geometry.PointCloud",
    method: str = "poisson",
    poisson_depth: int = 9,
    trim_density_quantile: float = 0.05,
    bpa_radii_factors: Optional[List[float]] = None,
) -> "o3d.geometry.TriangleMesh":
    """Main surface reconstruction entrypoint."""
    if method.lower() == "poisson":
        return reconstruct_surface_poisson(
            pcd, depth=poisson_depth, trim_density_quantile=trim_density_quantile
        )
    elif method.lower() == "bpa":
        return reconstruct_surface_bpa(pcd, radii_factors=bpa_radii_factors)
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")
