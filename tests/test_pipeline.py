import unittest
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

if HAS_OPEN3D:
    from src.synthetic import generate_synthetic_scene
    from src.preprocessing import preprocess_point_cloud, estimate_normals, remove_outliers
    from src.registration import register_point_clouds
    from src.reconstruction import reconstruct_surface, clean_mesh


@unittest.skipUnless(HAS_OPEN3D, "Open3D is required for pipeline unit tests.")
class TestPointCloudPipeline(unittest.TestCase):

    def setUp(self):
        self.scans, self.gt_poses = generate_synthetic_scene(
            num_views=2,
            points_per_view=1500,
            noise_std=0.002,
            outlier_ratio=0.02,
            random_seed=123,
        )

    def test_synthetic_generation(self):
        self.assertEqual(len(self.scans), 2)
        self.assertGreater(len(self.scans[0].points), 1000)

    def test_outlier_removal(self):
        raw_pcd = self.scans[0]
        cleaned = remove_outliers(raw_pcd, method="sor", nb_neighbors=15, std_ratio=1.8)
        self.assertLessEqual(len(cleaned.points), len(raw_pcd.points))

    def test_normal_estimation(self):
        pcd = self.scans[0]
        pcd = estimate_normals(pcd, radius=0.05, max_nn=20)
        self.assertTrue(pcd.has_normals())
        normals = np.asarray(pcd.normals)
        norms = np.linalg.norm(normals, axis=1)
        # Verify normals are normalized unit vectors
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)

    def test_pairwise_registration(self):
        pcd0 = preprocess_point_cloud(self.scans[0], voxel_size=0.03)
        pcd1 = preprocess_point_cloud(self.scans[1], voxel_size=0.03)

        trans, fitness, inlier_rmse = register_point_clouds(
            pcd1, pcd0, voxel_size=0.03, method="point_to_plane"
        )
        self.assertEqual(trans.shape, (4, 4))
        self.assertGreater(fitness, 0.2)
        self.assertLess(inlier_rmse, 0.05)

    def test_reconstruction(self):
        pcd0 = preprocess_point_cloud(self.scans[0], voxel_size=0.03)
        mesh = reconstruct_surface(pcd0, method="poisson", poisson_depth=6)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.triangles), 0)


if __name__ == "__main__":
    unittest.main()
