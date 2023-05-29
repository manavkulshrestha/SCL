import numpy as np
import open3d as o3d
from scipy.linalg import svd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw([inlier_cloud, outlier_cloud])


import numpy as np


def o3d_pcd(xyz: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    print(type(pcd))
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd


def fit_transform(src, dst):
    assert len(src) == len(dst)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_pcd(src), o3d_pcd(dst), 0.002, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return reg_p2p.transformation


def main():
    pcd = o3d.io.read_point_cloud("good.ply")
    print(pcd)
    print(np.asarray(pcd.points))

    cl, ind = pcd.remove_radius_outlier(nb_points=15, radius=0.01)
    display_inlier_outlier(pcd, ind)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(cl)
    vis.run()  # user picks points
    vis.destroy_window()

    # #298041 (0.086, 0.18, -1.5)
    # #213090 (0.11, 0.36, -1.5)
    # #222582 (0.29, 0.34, -1.5)


if __name__ == '__main__':
    main()
