# Multi-View Point Cloud Processing & Surface Reconstruction

A modular Python/Open3D pipeline for preprocessing, aligning, and reconstructing 3D surface meshes from multi-view point cloud scans.

Built for multi-camera and sensor capture setups (photogrammetry, depth sensors, LiDAR), handling noisy raw observations, computing coarse-to-fine spatial registrations, and generating watertight triangle meshes.

---

## Pipeline Overview

```
Raw Multi-View Scans (.ply / .pcd)
               │
               ▼
┌───────────────────────────────┐
│     Spatial Preprocessing     │  --> Voxel downsampling (octree grid)
│                               │  --> Statistical Outlier Removal (SOR)
│                               │  --> Radius Outlier Removal (ROR)
│                               │  --> Covariance-based normal estimation
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│     Multi-View Registration   │  --> FPFH (Fast Point Feature Histograms)
│                               │  --> Global coarse alignment (RANSAC)
│                               │  --> Point-to-Plane ICP refinement
│                               │  --> Pose Graph Optimization (loop closure)
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│       Dense Scan Fusion       │  --> Unified SE(3) transformation
│                               │  --> Consistent tangent plane normal reorientation
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│     Surface Reconstruction    │  --> Screened Poisson Reconstruction / BPA
│                               │  --> Density-based boundary face trimming
│                               │  --> Degenerate & non-manifold cleanup
└──────────────┬────────────────┘
               │
               ▼
Watertight Surface Mesh (.ply / .obj)
```

---

## Key Features

- **Robust Outlier Removal**: Combines $k$-NN local distance distribution filtering (SOR) with spatial radius clustering (ROR) to isolate sensor artifacts and airborne noise.
- **Normal Estimation & Alignment**: Uses local principal component analysis (PCA) with hybrid KD-Tree search, followed by viewpoint-directed and tangent plane propagation for coherent surface orientation.
- **Two-Stage Registration**:
  - *Coarse*: 33-dimensional FPFH feature descriptors with edge-length verified RANSAC matching.
  - *Fine*: Linearized Point-to-Plane ICP minimizing orthogonal point-to-tangent distances.
- **Global Pose Graph Optimization**: Employs Levenberg-Marquardt non-linear least squares across odometry and loop-closure edges for globally consistent multi-view alignment without drift.
- **Surface Reconstruction**:
  - *Screened Poisson Surface Reconstruction*: Solves an indicator function $\nabla \cdot \nabla \chi = \nabla \cdot \vec{V}$ with octree depth control and density-based boundary trimming.
  - *Ball-Pivoting Algorithm (BPA)*: Multi-scale sphere rolling for non-interpolated raw vertex preservation.
- **Self-Contained Benchmark Generator**: Includes synthetic multi-angle scanner simulation with sensor noise, occlusion, and arbitrary SE(3) poses for reproducible testing.

---

## Installation

```bash
git clone https://github.com/a-elrawy/multiview-pointcloud-pipeline.git
cd multiview-pointcloud-pipeline

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quickstart

### 1. Run Demo on Synthetic Benchmark
Run an end-to-end test without downloading external datasets:

```bash
python demo.py
```
This generates 4 synthetic multi-view scans, registers their poses, extracts a watertight Poisson mesh, and saves outputs to `outputs/demo_results/`.

### 2. Process Custom Scans via CLI
Process a directory of `.ply` or `.pcd` scans:

```bash
python run_pipeline.py --input_dir /path/to/scans --output_dir outputs/run1 --config configs/default_config.yaml
```

CLI options:
```text
  --input_dir PATH      Directory containing input scans (.ply, .pcd)
  --output_dir PATH     Output folder for point clouds and meshes (default: outputs)
  --config PATH         YAML pipeline config (default: configs/default_config.yaml)
  --demo                Run pipeline with synthetic multi-view data
  --views INT           Number of synthetic camera views (default: 4)
```

---

## Configuration

Tune pipeline parameters directly in `configs/default_config.yaml`:

```yaml
preprocessing:
  voxel_size: 0.02
  sor:
    enabled: true
    nb_neighbors: 20
    std_ratio: 2.0
  normal_estimation:
    radius: 0.05
    max_nn: 30

registration:
  method: "point_to_plane" # "point_to_plane" or "point_to_point"
  voxel_size: 0.02
  icp:
    max_correspondence_distance: 0.03
    max_iteration: 50

reconstruction:
  method: "poisson" # "poisson" or "bpa"
  poisson:
    depth: 9
    trim_density_quantile: 0.05
```

---

## Repository Structure

```
multiview-pointcloud-pipeline/
├── configs/
│   └── default_config.yaml      # Preprocessing, ICP, and Poisson parameters
├── src/
│   ├── __init__.py
│   ├── io.py                    # Point cloud and triangle mesh I/O
│   ├── preprocessing.py         # SOR/ROR filters, voxelization, normals
│   ├── registration.py          # FPFH features, RANSAC, Point-to-Plane ICP, Pose Graph
│   ├── reconstruction.py        # Poisson & BPA surface reconstruction, mesh cleanup
│   ├── synthetic.py             # Multi-view scan generator for testing
│   └── pipeline.py              # End-to-end pipeline orchestrator
├── tests/
│   └── test_pipeline.py         # Unit tests
├── demo.py                      # One-command quickstart script
├── run_pipeline.py              # CLI entrypoint
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
