import pickle

import pybullet as p
import pybullet_data

import os.path as osp
import time
from itertools import zip_longest

from matplotlib import pyplot as plt
from toposort import toposort
from toposort import CircularDependencyError

import torch
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.transform import Rotation as R

from Datasets.dutility import get_alldataloaders, ADPATH, get_scenesdataloader
from Generation.gen_lib import simulate_scene_pc, Camera, PBObjectLoader, pc_reg_merge
from dataset_analysis import condition
from nn.Network import ObjectNet, DNet
from nn.PositionalEncoder import PositionalEncoding
from robot.robot import UR5
from rp import setup_basic, dep_dict, scene_graph, dep_graph, remove_objects
from utility import load_model, dist_q, dist_e, jaccard, mean, quat_angle, tid_name, visualize, make_pcd, tid_colors, \
    std, draw_sphere_marker


def setup_field_fromdata(node_ids, oid_tid, poss, orns, slow=False):
    """ takes the goal poss and orns and lays out the involved objects in a grid on the plane """
    y_range = [0.2, 0.5]
    x_range = [.2, -.2]
    loader = PBObjectLoader('../Generation/urdfc')

    num_objs = len(node_ids)
    g_oids = np.array(sorted(list(oid_tid.keys())))

    idx = 0
    for xpos in np.linspace(*x_range, 5):
        for ypos in np.linspace(*y_range, 5):
            if idx >= num_objs:
                break

            oid = g_oids[idx]  # goal_id order is same as curr_id order
            typ = tid_name(oid_tid[oid])
            pos, orn = poss[idx], orns[idx]

            # modify orn on xy plane
            # new_orn = orn
            rot = R.from_rotvec([0, 0, np.random.uniform(-np.pi / 2, np.pi / 2)])
            new_orn = (rot * R.from_quat(orn)).as_quat()

            c_oid = loader.load_obj(otype=typ, pos=(xpos, ypos, 0.01), quat=new_orn, wait=100, slow=slow)
            # c_oid = loader.load_obj(otype=typ, pos=poss[idx], quat=orns[idx], wait=100, slow=slow)
            p.changeDynamics(c_oid, -1, mass=0.05)
            idx += 1

    return loader


def planning(dep_g):
    try:
        return list(toposort(dep_dict(dep_g)))
    except CircularDependencyError as e:
        print(e)
        return None


def setup_env(oid_tid, node_ids, g_poss, g_orns, headless=False):
    # add plane and camera and stuff
    setup_basic(headless=headless)

    # set up workspace for rearrangement
    curr_state = setup_field_fromdata(node_ids, oid_tid, g_poss, g_orns)
    return curr_state


def get_current_features(feat_model, pos_enc, node_ids, *, goal_feats, i):
    y_range = [0.2, 0.5]
    x_range = [.2, -.2]

    center = np.array([mean(x_range), mean(y_range), 0])
    cams_pos = [
        np.add([0.2, 0.2, .4], [0.05, -0.05, 0]),
        np.add([0.2, 0.5, .4], [0.05, 0.05, 0]),
        np.add([*center[:2], .4], [0, 0, 0])
    ]

    # draw_sphere_marker(cams_pos[0], color=[1, 0, 0, 1])
    # draw_sphere_marker(cams_pos[1], color=[0, 1, 0, 1])
    # draw_sphere_marker(cams_pos[2], color=[0, 0, 1, 1])

    cams = [Camera(pos, target=center) for pos in cams_pos]
    # i = 9237 is error
    unpack_hstack = lambda pci, coi: np.hstack([pci, coi.reshape(-1, 1)])
    pcti = np.concatenate([unpack_hstack(*cam.get_point_cloud()) for cam in cams])
    pcds, o_ids = pc_reg_merge(pcti[:, :3], pcti[:, -1])

    nodes_feats = []

    obj_id_unique = node_ids  # sorted(np.unique(o_ids))
    for oid, goal_feat in zip(obj_id_unique, goal_feats):  # NODE IDS is GOAL NODE IDS, OID is current scene oid
        obj_pcd = pcds[oid == o_ids]
        obj_cen = obj_pcd.mean(axis=0)

        try:
            sample_count = 512
            idx = np.random.choice(len(obj_pcd), sample_count, replace=len(obj_pcd) < sample_count)
            obj_pcd = obj_pcd[idx]
        except Exception as e:
            print(f'[{i}] {e}')
            return None

        # get total features from positional encoding of centroid and object level features from network
        cen_ten = torch.tensor(obj_cen, dtype=torch.float).cuda()
        pred_tid, obj_emb = feat_model.embed(obj_pcd, get_pred=True)

        # goal_tid = feat_model.predict_fromfeatures(goal_feat[255:])
        # print(f'VISUALIZING WHAT IS A {tid_name(pred_tid)}')
        # visualize(make_pcd(obj_pcd, colors=tid_colors(pred_tid)))

        obj_emb = torch.squeeze(obj_emb)
        pos_emb = pos_enc(cen_ten)
        x = torch.cat([pos_emb, obj_emb])
        nodes_feats.append(x)

    return torch.stack(nodes_feats)


def main():
    # load models
    feat_net = load_model(ObjectNet, 'cn_test_best_model.pt')
    feat_net.eval()

    pos_enc = PositionalEncoding(min_deg=0, max_deg=5, scale=1, offset=0)

    # load data
    _, _, test_loader = get_scenesdataloader(feat_net)
    print('done loading')

    cur_seg = (0, 10)
    data_counts = {cur_seg: 0}
    max_examples = 99999

    cur_count = 0

    base_data = []

    for i, data in enumerate(test_loader):  # 9196
        lb, ub = cur_seg
        nnum_objs = len(data.adj_mat[0])
        if not (lb < nnum_objs <= ub):
            print(f'[{i}] skipping because {nnum_objs} objs')
            continue
        if data_counts[cur_seg] >= max_examples:
            print(f'[{i}] skipping because already have {data_counts[cur_seg]} >= {max_examples} samples')
            continue
        curr_state = setup_env(data.oid_tid[0][0], data.node_ids[0],
                               data.g_poss[0], data.g_orns[0], headless=True)
        plan = planning(data.adj_mat[0])
        curr_x = get_current_features(feat_net, pos_enc, data.node_ids[0], goal_feats=data.x, i=i)
        p.disconnect()
        if curr_x is None or plan is None:
            continue

        data_counts[cur_seg] += 1
        cur_count = sum(data_counts.values())
        print(f'done with: i={i}, cur_count={cur_count}')

        this_scene_data = {
            'sceneloader_i': i,
            'goal_obj_ids': data.node_ids,
            'goal_x': data.x,
            'init_x': curr_x,
            'plan': plan,
            'oid_tid': data.oid_tid[0][0],
            'node_ids': data.node_ids[0],
            'adj_mat': data.adj_mat[0],
            'g_poss': data.g_poss[0],
            'g_orns': data.g_orns[0]
        }

        base_data.append(this_scene_data)

    with open(f'data/baselinedata_{cur_count}_{cur_seg[0]}-{cur_seg[1]}-{int(time.time())}', 'wb') as f:
        pickle.dump(base_data, f)


if __name__ == '__main__':
    main()
