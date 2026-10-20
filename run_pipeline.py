#!/usr/bin/env python3
"""CLI runner for Multi-View Point Cloud Reconstruction Pipeline."""

import os
import glob
import argparse
import logging
import yaml

from src.io import load_point_cloud, save_point_cloud, save_mesh
from src.synthetic import generate_synthetic_scene
from src.pipeline import MultiViewReconstructionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_runner")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-View Point Cloud Processing & Surface Reconstruction"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input point cloud scans (.ply, .pcd).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save reconstructed point clouds and meshes.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run end-to-end demo using synthetically generated multi-view scans.",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=4,
        help="Number of synthetic camera views to simulate (only used with --demo).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded configuration from %s", args.config)
    else:
        logger.warning("Config file %s not found, using system defaults.", args.config)

    scans = []
    if args.demo or not args.input_dir:
        logger.info("Running synthetic multi-view benchmark scene (%d views)...", args.views)
        scans, _ = generate_synthetic_scene(num_views=args.views)
        # Optionally save raw scans for inspection
        raw_dir = os.path.join(args.output_dir, "raw_scans")
        for i, s in enumerate(scans):
            save_point_cloud(s, os.path.join(raw_dir, f"scan_{i:02d}.ply"))
    else:
        file_patterns = [
            os.path.join(args.input_dir, "*.ply"),
            os.path.join(args.input_dir, "*.pcd"),
        ]
        files = []
        for p in file_patterns:
            files.extend(glob.glob(p))
        files.sort()

        if not files:
            raise FileNotFoundError(f"No point cloud files found in {args.input_dir}")

        logger.info("Found %d scan files in %s", len(files), args.input_dir)
        scans = [load_point_cloud(f) for f in files]

    pipeline = MultiViewReconstructionPipeline(config=config)
    results = pipeline.process(scans)

    # Save output artifacts
    pcd_out = os.path.join(args.output_dir, "fused_dense_cloud.ply")
    mesh_out = os.path.join(args.output_dir, "reconstructed_mesh.ply")
    mesh_obj_out = os.path.join(args.output_dir, "reconstructed_mesh.obj")

    save_point_cloud(results["fused_point_cloud"], pcd_out)
    save_mesh(results["reconstructed_mesh"], mesh_out)
    save_mesh(results["reconstructed_mesh"], mesh_obj_out)

    logger.info("==========================================")
    logger.info("Reconstruction successfully completed!")
    logger.info("Output point cloud: %s", pcd_out)
    logger.info("Output mesh:        %s", mesh_out)
    logger.info("==========================================")


if __name__ == "__main__":
    main()
