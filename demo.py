#!/usr/bin/env python3
"""Quickstart demonstration script.

Generates a synthetic multi-view scan set, executes registration and
Screened Poisson surface reconstruction, and exports the resulting watertight 3D mesh.
"""

import os
import logging
from src.synthetic import generate_synthetic_scene
from src.pipeline import MultiViewReconstructionPipeline
from src.io import save_point_cloud, save_mesh

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("demo")


def run():
    output_dir = "outputs/demo_results"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- 1. Generating Multi-View Synthetic Scans ---")
    scans, ground_truth_poses = generate_synthetic_scene(
        num_views=4,
        points_per_view=4000,
        noise_std=0.003,
        outlier_ratio=0.03
    )

    print("\n--- 2. Executing Multi-View Pipeline ---")
    pipeline = MultiViewReconstructionPipeline()
    results = pipeline.process(scans)

    fused_pcd = results["fused_point_cloud"]
    mesh = results["reconstructed_mesh"]

    print("\n--- 3. Exporting 3D Artifacts ---")
    pcd_path = os.path.join(output_dir, "fused_scene.ply")
    mesh_path = os.path.join(output_dir, "surface_mesh.ply")
    mesh_obj = os.path.join(output_dir, "surface_mesh.obj")

    save_point_cloud(fused_pcd, pcd_path)
    save_mesh(mesh, mesh_path)
    save_mesh(mesh, mesh_obj)

    print("\n[SUCCESS] Pipeline completed successfully!")
    print(f"Fused Point Cloud: {pcd_path} ({len(fused_pcd.points)} points)")
    print(f"Watertight Mesh:   {mesh_path} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)\n")


if __name__ == "__main__":
    run()
