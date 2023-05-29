import numpy as np
import open3d as o3d


# def o3d_pcd(xyz: np.ndarray) -> o3d.cuda.pybind.geometry.PointCloud:
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#
#     return pcd
#
#
# def apply_transform(T, xyz):
#     return np.array(o3d_pcd(xyz).transform(T).points)
#
#
# def fit_transform(src, dst):
#     assert len(src) == len(dst)
#
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         o3d_pcd(src), o3d_pcd(dst), 0.02, np.eye(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     )
#
#     return reg_p2p.transformation


def fit_transform(src, dst):
    s_cen = src.mean(axis=0)
    d_cen = dst.mean(axis=0)

    # P = np.mean([np.outer(sk, sk) for sk in src], axis=0) - np.outer(s_cen, s_cen)
    # Q = np.mean([np.outer(dk, sk) for dk, sk in zip(dst, src)], axis=0) - np.outer(d_cen, s_cen)

    P = src.T @ src - np.outer(s_cen, s_cen)
    Q = dst.T @ src - np.outer(d_cen, s_cen)

    return Q, np.linalg.pinv(P), s_cen, d_cen


def apply_transform(tr: tuple, src: np.ndarray) -> np.ndarray:
    Q, Pinv, s_cen, d_cen = tr
    return (Q @ Pinv @ (src - s_cen).T).T + d_cen


def main():

    # #298041 (0.086, 0.18, -1.5)
    # #213090 (0.11, 0.36, -1.5)
    # #222582 (0.29, 0.34, -1.5)

    cam_points = np.array([(0.086, 0.18, -1.5), (0.11, 0.36, -1.5), (0.29, 0.34, -1.5)])
    c0, c1, c2 = [cc for cc in cam_points]
    s0 = [-0.010150188246474658, 0.10913780113131626, 0.008155820746643228]
    s1 = [0.1872985578980705, 0.10915033057458054, 0.011853772508362753]
    s2 = [0.1804238344673205, -0.08996453240191944, 0.010479755950644945]
    sim_points = np.array([s0, s1, s2])

    src = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dst = np.array([[2.5, 3.5, 4.5], [5.5, 6.5, 7.5], [8.5, 9.5, 10.5]])
    src = cam_points
    dst = sim_points

    tr = fit_transform(src, dst)
    img = apply_transform(tr, src)

    print(f'INPUT = \n\{src}n')

    print(img, '\n', dst)
    print(np.allclose(img, dst))

    print(tr)
    print(apply_transform(tr, np.array([(0.37, 0.028, -1.4)])))


if __name__ == '__main__':
    main()
