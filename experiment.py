import pybullet as p
import pybullet_data

import os.path as osp
import time
from itertools import zip_longest

from matplotlib import pyplot as plt
from toposort import toposort

import torch
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.transform import Rotation as R

from Datasets.dutility import get_alldataloaders, ADPATH
from Generation.gen_lib import simulate_scene_pc, Camera, PBObjectLoader
from nn.Network import ObjectNet, DNet
from robot.robot import UR5
from rp import setup_basic, dep_dict, scene_graph, dep_graph, remove_objects
from utility import load_model, dist_q, dist_e, jaccard, mean, quat_angle, tid_name, visualize, make_pcd, tid_colors


def setup_field_fromdata(node_ids, o_ids, t_ids, poss, orns, slow=False):
    """ takes the goal poss and orns and lays out the involved objects in a grid on the plane """
    x_range = [-0.2, -0.5]
    y_range = [-.2, .2]
    loader = PBObjectLoader('Generation/urdfc')

    num_objs = len(node_ids)
    oid_tid = dict(zip(o_ids, t_ids))
    g_oids = np.unique(o_ids)

    idx = 0
    for xpos in np.linspace(*x_range, 5):
        for ypos in np.linspace(*y_range, 5):
            if idx >= num_objs:
                break

            oid = g_oids[idx]
            typ = tid_name(oid_tid[oid])
            pos, orn = poss[idx], orns[idx]

            # modify orn on xy plane
            rot = R.from_rotvec([0, 0, np.random.uniform(0, 2 * np.pi)])
            new_orn = (rot * R.from_quat(orn)).as_quat()

            c_oid = loader.load_obj(otype=typ, pos=(xpos, ypos, 0.01), quat=new_orn, wait=100, slow=slow)
            p.changeDynamics(c_oid, -1, mass=0.05)
            idx += 1

    return loader, g_oids


def get_target_pose(idx, poss, orns):
    return poss[idx], orns[idx]


def rearrangement_metrics(moved_idx, curr_state, g_poss, g_orns):
    pos_err, orn_err, ora_err = [], [], []
    for obj_idx in moved_idx:
        c_oid = curr_state.obj_ids[obj_idx]

        g_pos, g_orn = g_poss[obj_idx], g_orns[obj_idx]
        c_pos, c_orn = p.getBasePositionAndOrientation(c_oid)

        pos_err.append(dist_e(c_pos, g_pos))
        orn_err.append(dist_q(c_orn, g_orn))
        ora_err.append(quat_angle(c_orn, g_orn))

    # report metrics
    pos_err = np.array(pos_err)
    orn_err = np.array(orn_err)
    ora_err = np.array(ora_err)
    # print(pos_err)
    # print(orn_err)
    orn_err /= np.sqrt(2)

    # print(f'pos error: {np.mean(pos_err):.4f}+/-{np.std(pos_err):.4f}, max: {np.max(pos_err):.4f}, min: {np.min(pos_err):.4f}')
    # print(f'orn error: {np.mean(orn_err):.4f}+/-{np.std(orn_err):.4f}, max: {np.max(orn_err):.4f}, min: {np.min(orn_err):.4f}')
    # print(f'ora error: {np.mean(ora_err):.4f}+/-{np.std(ora_err):.4f}, max: {np.max(ora_err):.4f}, min: {np.min(ora_err):.4f}')
    # print(f'planning time: {planning_time:.6f} (dependence graph) and {withsorting_time:.6f} (with sorting)')

    return {
        'pos_err': pos_err, 'pos_std': np.std(pos_err), 'pos_min': np.min(pos_err), 'pos_max': np.max(pos_err),
        'orn_err': orn_err, 'orn_std': np.std(orn_err), 'orn_min': np.min(orn_err), 'orn_max': np.max(orn_err),
        'ora_err': ora_err, 'ora_std': np.std(ora_err), 'ora_min': np.min(ora_err), 'ora_max': np.max(ora_err),
    }


def planning(node_graph, dep_net, dep_g):
    # infer scene structure/planning
    start = time.time()
    pred_graph = scene_graph(node_graph, dep_model=dep_net)  # TODO add adaptation to reduce threshold
    inference_time = time.time() - start
    pred_layers = list(toposort(dep_dict(pred_graph)))
    planning_time = time.time() - start
    gt_graph = dep_g

    # get jaccard similarity of layers # list(toposort(dep_dict(gt_graph)))
    gt_layers = list(toposort(dep_dict(gt_graph)))
    layer_jaccards = [jaccard(gt_l, pr_l) for gt_l, pr_l in zip_longest(list(gt_layers), list(pred_layers),
                                                                        fillvalue=set())]
    mean_jaccard = mean(layer_jaccards)

    return pred_layers, {
        'inference_time': inference_time,
        'planning_time': planning_time,
        'layer_jaccards': layer_jaccards,
        'mean_jaccard': mean_jaccard
    }


def rearrangement(robot, pred_layers, curr_state, poss, orns, timeout=20000):
    start_time = time.time()

    moved_idx = []
    for layer in pred_layers:
        for obj_idx in layer:
            moved_idx.append(obj_idx)
            c_oid = curr_state.obj_ids[obj_idx]

            # in real, position obtained from point-cloud and orientation from TEASER++
            c_pos_cen, c_orn = p.getBasePositionAndOrientation(c_oid)
            g_pos_cen, g_orn = get_target_pose(obj_idx, poss, orns)
            c_pos = np.array(c_pos_cen)
            g_pos_cen = np.array(g_pos_cen)

            # obtain goal orientation
            g_orn_to = p.getDifferenceQuaternion(c_orn, g_orn)  # in real, done with TEASER++

            # move above cur position, move to curr, pick, move above curr
            robot.move_ee_above(c_pos, orn=(0, 0, 0, 1))
            c_pos_from, _ = robot.move_ee_down(c_pos, orn=(0, 0, 0, 1))
            robot.suction(True)
            robot.move_ee_above(c_pos, orn=(0, 0, 0, 1))

            # obtain goal pose
            succ_offt = np.subtract(c_pos_from, c_pos_cen)
            g_orn_mat = R.from_quat(g_orn_to).as_matrix()
            g_pos_to = g_pos_cen + (g_orn_mat @ succ_offt)

            # move above goal position, move to goal, drop, move above goal
            robot.move_ee_above(g_pos_to, orn=g_orn_to)
            robot.move_ee(g_pos_to + [0, 0, 0.005], orn=g_orn_to)
            robot.suction(False)

            for _ in range(500):
                p.stepSimulation()

            robot.move_ee_above(g_pos_cen, orn=(0, 0, 0, 1))

            if (time.time() - start_time) > timeout:
                return False, moved_idx, rearrangement_metrics(moved_idx, curr_state, poss, orns)

    return True, moved_idx, rearrangement_metrics(moved_idx, curr_state, poss, orns)


def setup_env(o_ids, t_ids, node_ids, poss, orns, headless=False):
    # add plane and camera and stuff
    setup_basic(headless=headless)

    # set up workspace for rearrangement
    curr_state, g_oids = setup_field_fromdata(node_ids, o_ids, t_ids, poss, orns)
    robot = UR5([-0.5, 0, 0])
    for _ in range(100):
        p.stepSimulation()

    return robot, curr_state, g_oids


def recreate_scene(scene_num):
    scene_name = f'{scene_num // 1000}_{scene_num % 1000}.npz'
    all_path = osp.join(ADPATH, scene_name)

    all_file = np.load(all_path)
    to_extract = ['pc', 'oid', 'tid', 'depg', 'pos', 'orn', 'node_ids']
    pcds, o_ids, t_ids, dep_g, g_poss, g_orns, node_ids = [all_file[x] for x in to_extract]

    setup_basic()
    loader = PBObjectLoader('Generation/urdfc')
    oid_tid = dict(zip(o_ids, t_ids.astype(int)))

    visualize(make_pcd(pcds, tid_colors(t_ids)))

    for g_oid, g_pos, g_orn in zip(np.unique(o_ids).astype(int), g_poss, g_orns):
        tid = oid_tid[g_oid]
        typ = tid_name(tid)
        loader.load_obj(typ, g_pos, g_orn)


    return loader


def main():
    # load models
    feat_net = load_model(ObjectNet, 'cn_test_best_model.pt')
    feat_net.eval()

    dep_net = load_model(DNet, 'dnT_best_model_95_nn.pt',
                         model_args=[511, 256, 128], model_kwargs={'heads': 16, 'concat': False})
    dep_net.eval()

    # load data and do experiments
    _, _, test_loader = get_alldataloaders(feat_net)
    for i, data in enumerate(test_loader):
        robot, initial_state, g_oids = setup_env(data.o_ids[0], data.t_ids[0], data.node_ids[0],
                                                 data.g_poss[0], data.g_orns[0])
        robot.move_timestep = 0
        pred_layers, p_metrics = planning(data, dep_net, data.adj_mat[0])
        timeout, moved_idx, r_metrics = rearrangement(robot, pred_layers, initial_state, data.g_poss[0], data.g_orns[0])
        p.disconnect()

        print('success:', not timeout)
        # TODO log {'moved_idx': moved_idx, **p_metrics, **r_metrics, 'num_nodes': len(data.dep_g)}

    # TODO completion, logging in file, analysis/readout script

    # recreate_scene(9000)
    # time.sleep(100)


if __name__ == '__main__':
    main()
