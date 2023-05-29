import time
from typing import Iterable

import numpy as np
import open3d as o3d
from open3d.examples.geometry.point_cloud_outlier_removal_radius import display_inlier_outlier

from Generation.gen_lib import PBObjectLoader
from robot.robot import UR5

import pybullet as p
import pybullet_data

up_q = np.array([-1, -0.5, 0, -0.5, -0.5, 0]) * np.pi
home_q = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi


def draw_sphere_marker(position, radius=0.02, color=(0, 0, 1, 1)):
    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
    return marker_id


def sim_setup():
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    target = (-0.07796166092157364, 0.005451506469398737, -0.06238798052072525)
    dist = 1.0
    yaw = 89.6000747680664
    pitch = -17.800016403198242
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")

    robot_offt_z = -np.mean(
        [-0.1867676817112838,
         -0.18596259913093333,
         -0.1902045026366347,
         -0.19100720846428204,
         -0.191706796360579]
    )
    ur5 = UR5([-0.5, 0, robot_offt_z + 0.034])
    ur5.set_q(ur5.home_q)
    for _ in range(100):
        p.stepSimulation()

    return ur5


def fit_transform(src, dst):
    s_cen = src.mean(axis=0)
    d_cen = dst.mean(axis=0)

    P = src.T @ src - np.outer(s_cen, s_cen)
    Q = dst.T @ src - np.outer(d_cen, s_cen)

    return Q, np.linalg.pinv(P), s_cen, d_cen


def apply_transform(tr: tuple, src: np.ndarray) -> np.ndarray:
    Q, Pinv, s_cen, d_cen = tr
    return (Q @ Pinv @ (src - s_cen).T).T + d_cen


def pcd_selection(pcd_file, use_saved=False):
    pcd_old = o3d.io.read_point_cloud("good.ply")
    pcd_np = np.array(pcd_old.points)
    pcd, ind = pcd_old.remove_radius_outlier(nb_points=15, radius=0.01)
    # display_inlier_outlier(pcd, ind)

    if not use_saved:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()

        selected_idx = vis.get_picked_points()
        selected_np = np.array(pcd.select_by_index(selected_idx).points)
        np.savez(f'{pcd_file.split(".")[0]}.npz', selected_np=selected_np, selected_idx=selected_idx, pcd_np=pcd_np)
    else:
        save_file = np.load(f'{pcd_file.split(".")[0]}.npz')
        selected_np, selected_idx, pcd_np = [save_file[x] for x in ['selected_np', 'selected_idx', 'pcd_np']]

    sel_len = len(selected_np)
    print(selected_idx)
    # assert sel_len == 3+12, f'There should be 3 correspondence points elected and 12 object points. Found {sel_len}'
    return pcd, selected_np[:3], selected_np[3:]


def get_sim_correspondences(robot):
    # corr_q = [
    #     [-3.1416080633746546, -0.7186852258494874, 2.1041701475726526, -2.956341405908102, -1.5708392302142542,
    #           3.109525277977809e-05],
    #     [-3.141592089329855, -0.5305090707591553, 1.4884045759784144, -2.528691907922262, -1.5708068052874964,
    #       5.460591273731552e-05],
    #     [-3.4305806795703333, -0.5385233920863648, 1.5208080450641077, -2.552784582177633, -1.5708830992328089,
    #       -0.2889187971698206]
    # ]
    # sim_cor = []
    # for q in corr_q:
    #     robot.move_q(q)
    #     sim_cor.append(robot.ee_pose[0])
    # return np.array(sim_cor)

    s0 = [-0.010150188246474658, 0.10913780113131626, 0.008155820746643228]
    s1 = [0.1872985578980705, 0.10915033057458054, 0.011853772508362753]
    s2 = [0.1804238344673205, -0.08996453240191944, 0.010479755950644945]
    return np.array([s0, s1, s2])


def main():
    robot = sim_setup()
    robot.set_q(up_q)
    for _ in range(100):
        p.stepSimulation()

    obj_tna = ['pyramid', 'cube', 'cylinder',
               'ccuboid', 'cube', 'cylinder',
               'ccuboid', 'cube', 'cube',
               'cuboid', 'cuboid', 'roof']
    init_orn = lambda tna: (0, 0, -np.pi/2) if tna == 'ccuboid' else (0, 0, 0)

    pcd_file = 'good.ply'
    pcd, pcd_cor, pcd_pts = pcd_selection(pcd_file, use_saved=False)
    sim_cor = get_sim_correspondences(robot)

    pcd2sim = fit_transform(pcd_cor, sim_cor)
    # pcd2sim = (np.array([[0.05274394, 0.09198194, -0.3575722],
    #         [-0.0116268, 0.01580379, -0.1283236],
    #         [0.00339802, 0.00631734, -0.03048935]]), np.array([[54.01933448, -30.85761983, -0.20029087],
    #                                                         [-30.85761983, 68.99674996, 10.16007483],
    #                                                         [-0.20029087, 10.16007483, 2.18744989]]),
    #  np.array([0.162, 0.29333333, -1.5]), np.array([0.11919073, 0.04277453, 0.01016312]))

    sim_pts = apply_transform(pcd2sim, pcd_pts)
    pcd_cor_img = apply_transform(pcd2sim, pcd_cor)

    # loader = PBObjectLoader('../Generation/urdfc')
    # for pos, tna in zip(sim_pts, obj_tna):
    #     loader.load_obj(tna, pos, euler=init_orn(tna))
    for sp in sim_pts:
        draw_sphere_marker(sp)

    print(f'PCD correspondences = \n {pcd_cor}\n')

    print(f'Simulation correspondences:\n{sim_cor}\n')
    print(f'PCD correspondences image in sim space:\n{pcd_cor_img}\n')

    print(np.allclose(sim_cor, pcd_cor_img))
    print(pcd2sim)

    time.sleep(500)


if __name__ == '__main__':
    main()
