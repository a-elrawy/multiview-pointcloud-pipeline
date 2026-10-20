import logging
from typing import List, Tuple, Optional
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logger = logging.getLogger(__name__)


def compute_fpfh_features(
    pcd: "o3d.geometry.PointCloud",
    voxel_size: float,
    radius_multiplier: float = 5.0,
    max_nn: int = 100,
) -> "o3d.pipelines.registration.Feature":
    """Compute Fast Point Feature Histograms (FPFH) for global registration."""
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for feature computation.")

    radius_feature = voxel_size * radius_multiplier
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn),
    )
    return fpfh


def execute_global_registration(
    source_down: "o3d.geometry.PointCloud",
    target_down: "o3d.geometry.PointCloud",
    source_fpfh: "o3d.pipelines.registration.Feature",
    target_fpfh: "o3d.pipelines.registration.Feature",
    voxel_size: float,
    distance_multiplier: float = 1.5,
    max_iterations: int = 100000,
    confidence: float = 0.999,
) -> "o3d.pipelines.registration.RegistrationResult":
    """Coarse global registration using RANSAC based on FPFH feature matching."""
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for global registration.")

    distance_threshold = voxel_size * distance_multiplier
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iterations, confidence
        ),
    )
    return result


def execute_icp_registration(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    init_transformation: np.ndarray,
    max_correspondence_distance: float = 0.03,
    method: str = "point_to_plane",
    max_iteration: int = 50,
) -> "o3d.pipelines.registration.RegistrationResult":
    """Refine alignment using Iterative Closest Point (ICP).

    Supports Point-to-Plane (preferred for smooth surfaces with normals)
    or Point-to-Point.
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for ICP registration.")

    if method.lower() == "point_to_plane":
        estimation = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
    else:
        estimation = (
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iteration,
        relative_fitness=1e-6,
        relative_rmse=1e-6,
    )

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        init_transformation,
        estimation,
        criteria,
    )
    return result


def register_point_clouds(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    voxel_size: float = 0.02,
    method: str = "point_to_plane",
    max_correspondence_distance: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Pairwise registration pipeline:

    1. FPFH feature extraction on downsampled representations
    2. Coarse global RANSAC alignment
    3. Fine Point-to-Plane ICP refinement

    Returns:
        transformation: 4x4 rigid transformation matrix SE(3)
        fitness: Inlier correspondence ratio
        inlier_rmse: Root mean squared error of corresponding inliers
    """
    if max_correspondence_distance is None:
        max_correspondence_distance = voxel_size * 1.5

    # Downsample for fast feature matching
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    source_fpfh = compute_fpfh_features(source_down, voxel_size)
    target_fpfh = compute_fpfh_features(target_down, voxel_size)

    # Coarse registration
    coarse_result = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )

    # Fine registration
    fine_result = execute_icp_registration(
        source,
        target,
        init_transformation=coarse_result.transformation,
        max_correspondence_distance=max_correspondence_distance,
        method=method,
    )

    logger.debug(
        "Pairwise registration: fitness=%.4f, inlier_rmse=%.6f",
        fine_result.fitness,
        fine_result.inlier_rmse,
    )
    return fine_result.transformation, fine_result.fitness, fine_result.inlier_rmse


def multiway_registration(
    pcds: List["o3d.geometry.PointCloud"],
    voxel_size: float = 0.02,
    max_correspondence_distance: Optional[float] = None,
) -> List[np.ndarray]:
    """Register an arbitrary sequence of multi-view point clouds into a unified global frame

    using Pose Graph Optimization.
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for multiway registration.")

    if len(pcds) == 1:
        return [np.eye(4)]

    if max_correspondence_distance is None:
        max_correspondence_distance = voxel_size * 1.5

    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            trans, fitness, rmse = register_point_clouds(
                pcds[source_id],
                pcds[target_id],
                voxel_size=voxel_size,
                max_correspondence_distance=max_correspondence_distance,
            )

            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcds[source_id],
                pcds[target_id],
                max_correspondence_distance,
                trans,
            )

            if target_id == source_id + 1:
                # Sequential odometry edge
                odometry = np.dot(trans, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        trans,
                        info,
                        uncertain=False,
                    )
                )
            elif fitness > 0.3:
                # Loop closure / cross-view connection
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        trans,
                        info,
                        uncertain=True,
                    )
                )

    # Global optimization of pose graph
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    criteria = (
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        criteria,
        option,
    )

    poses = [node.pose for node in pose_graph.nodes]
    logger.info("Optimized pose graph with %d nodes and %d edges.", len(poses), len(pose_graph.edges))
    return poses
