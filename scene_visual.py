import time

import numpy as np
import pybullet as p
import open3d as o3d
import pybullet_data

from Datasets.dutility import ADPATH
from Generation.gen_lib import PBObjectLoader
import os.path as osp

from utility import make_pcd, visualize, tid_colors, tid_name

def setup_basic(headless=False):
    """ sets up the camera, adds gravity, and adds the plane """
    physics_client = p.connect(p.DIRECT if headless else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    target = (-0.07796166092157364, 0.005451506469398737, -0.06238798052072525)
    dist = 1.0
    yaw = 89.6000747680664
    pitch = -17.800016403198242
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    plane_id = p.loadURDF('plane.urdf')

    return physics_client, plane_id

def recreate_scene(scene_num):
    scene_name = f'{scene_num // 1000}_{scene_num % 1000}.npz'
    all_path = osp.join(ADPATH, scene_name)

    all_file = np.load(all_path)
    to_extract = ['pc', 'oid', 'tid', 'depg', 'pos', 'orn', 'node_ids']
    pcds, o_ids, t_ids, dep_g, g_poss, g_orns, node_ids = [all_file[x] for x in to_extract]

    setup_basic()
    loader = PBObjectLoader('../Generation/urdfc')
    oid_tid = dict(zip(o_ids, t_ids.astype(int)))

    # visualize(make_pcd(pcds, tid_colors(t_ids)))

    for g_oid, g_pos, g_orn in zip(np.unique(o_ids).astype(int), g_poss, g_orns):
        tid = oid_tid[g_oid]
        typ = tid_name(tid)
        loader.load_obj(typ, g_pos, g_orn)

    time.sleep(1000)
    return loader


def main():
    recreate_scene(8000)


if __name__ == '__main__':
    main()
