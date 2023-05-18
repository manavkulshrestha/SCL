import jsonpickle
import os.path as osp
import os

import numpy as np
import pybullet as p

from Generation.gen_lib import Camera, simulate_scene_pc, pc_reg_merge, oid_typeidx, dep_graph
import pybullet_data


def setup_basic():
    """ sets up the camera, adds gravity, and adds the plane """
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    target = (-0.07796166092157364, 0.005451506469398737, -0.06238798052072525)
    dist = 1.0
    yaw = 89.6000747680664
    pitch = -17.800016403198242
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    plane_id = p.loadURDF('plane.urdf')

    return physics_client, plane_id


def main():
    cam1 = Camera([0.15, 0.15, .2])
    cam2 = Camera([0.15, -0.15, .2])
    cam3 = Camera([0, 0, .3])
    cams = [cam1, cam2, cam3]

    # ppath = osp.abspath('../data/pcd_data/')
    # dpath = osp.abspath('../data/dep_data/')
    apath = osp.abspath('../../../rp/data/all_data/')

    setup_basic()

    np.random.seed(142)
    seeds = np.random.randint(0, 10000, size=10)
    for seed in seeds:
        np.random.seed(seed)
        print(f'Now using seed {seed}')
        aspath = osp.join(apath, str(seed))
        os.makedirs(aspath, exist_ok=True)

        for i in range(1000):
            print(f'SCENE {i}:')
            print('Simulating and obtaining point cloud...', end='')
            if (loaderpcco := simulate_scene_pc(cams)) is not None:
                loader, pc, co = loaderpcco
                print('Done')

                print('Merging point clouds from camera angles...', end='')
                pc_m, co_m = pc_reg_merge(pc, co)
                print('Done')

                print('Getting cannonical type ids from body ids...', end='')
                co_mt = oid_typeidx(loader, co_m)
                print('Done')

                print('Saving dep data, pose data, and point cloud to file...', end='')
                node_ids = loader.obj_ids
                depg = dep_graph(node_ids)
                obj_poses = [loader.obj_poses[nid] for nid in node_ids]
                positions = [posi for posi, orie in obj_poses]
                orientations = [orie for posi, orie in obj_poses]

                np.savez(osp.join(aspath, f'{i}.npz'),
                         pc=pc_m, oid=co_m, tid=co_mt,
                         node_ids=node_ids, depg=depg,
                         pos=positions, orn=orientations)
                print('Done\n')

                p.disconnect()

            else:
                pass

        print(f'FINISHED {seed}')
    print('DONE WITH ALL')


if __name__ == '__main__':
    main()