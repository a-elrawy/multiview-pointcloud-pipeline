import os
import copy
import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from .preprocessing import preprocess_point_cloud
from .registration import multiway_registration
from .reconstruction import reconstruct_surface
from .io import save_point_cloud, save_mesh

logger = logging.getLogger(__name__)


class MultiViewReconstructionPipeline:
    """End-to-end pipeline for multi-view point cloud preprocessing,

    pose registration, scan fusion, and watertight 3D surface reconstruction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._setup_defaults()

    def _setup_defaults(self):
        self.voxel_size = self.config.get("preprocessing", {}).get("voxel_size", 0.02)
        self.sor_enabled = self.config.get("preprocessing", {}).get("sor", {}).get("enabled", True)
        self.sor_nb = self.config.get("preprocessing", {}).get("sor", {}).get("nb_neighbors", 20)
        self.sor_std = self.config.get("preprocessing", {}).get("sor", {}).get("std_ratio", 2.0)

        self.reg_method = self.config.get("registration", {}).get("method", "point_to_plane")
        self.recon_method = self.config.get("reconstruction", {}).get("method", "poisson")
        self.poisson_depth = self.config.get("reconstruction", {}).get("poisson", {}).get("depth", 9)
        self.trim_quantile = self.config.get("reconstruction", {}).get("poisson", {}).get("trim_density_quantile", 0.05)

    def process(self, raw_scans: List["o3d.geometry.PointCloud"]) -> Dict[str, Any]:
        """Execute the full reconstruction pipeline on a list of point cloud scans."""
        if not HAS_OPEN3D:
            raise ImportError("open3d is required to execute the pipeline.")

        if not raw_scans:
            raise ValueError("No input point cloud scans provided.")

        logger.info("Starting MultiViewReconstructionPipeline on %d scans.", len(raw_scans))

        # 1. Preprocess and denoise each view independently
        preprocessed_scans = []
        for i, scan in enumerate(raw_scans):
            logger.info("Preprocessing scan %d/%d (initial points: %d)...", i + 1, len(raw_scans), len(scan.points))
            clean_scan = preprocess_point_cloud(
                copy.deepcopy(scan),
                voxel_size=self.voxel_size,
                sor_enabled=self.sor_enabled,
                sor_nb_neighbors=self.sor_nb,
                sor_std_ratio=self.sor_std,
                estimate_normal=True,
            )
            preprocessed_scans.append(clean_scan)

        # 2. Multi-view registration & pose graph optimization
        logger.info("Registering multi-view scans via pose graph optimization...")
        poses = multiway_registration(
            preprocessed_scans,
            voxel_size=self.voxel_size,
        )

        # 3. Transform and merge into unified coordinate frame
        logger.info("Fusing scans into unified global representation...")
        fused_pcd = o3d.geometry.PointCloud()
        for scan, pose in zip(preprocessed_scans, poses):
            aligned = copy.deepcopy(scan)
            aligned.transform(pose)
            fused_pcd += aligned

        logger.info("Fused point cloud contains %d points.", len(fused_pcd.points))

        # Clean up and recompute consistent global normals
        fused_pcd = fused_pcd.voxel_down_sample(voxel_size=self.voxel_size * 0.8)
        fused_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2.5, max_nn=30
            )
        )
        fused_pcd.orient_normals_consistent_tangent_plane(k=15)

        # 4. 3D Surface Reconstruction
        logger.info("Executing surface reconstruction using [%s]...", self.recon_method.upper())
        mesh = reconstruct_surface(
            fused_pcd,
            method=self.recon_method,
            poisson_depth=self.poisson_depth,
            trim_density_quantile=self.trim_quantile,
        )

        return {
            "fused_point_cloud": fused_pcd,
            "reconstructed_mesh": mesh,
            "poses": poses,
            "num_scans": len(raw_scans),
        }
