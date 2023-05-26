import pickle
from typing import Iterable

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

from Datasets.dutility import get_alldataloaders, ADPATH, get_scenesdataloader
from Generation.gen_lib import simulate_scene_pc, Camera, PBObjectLoader
from dataset_analysis import condition
from nn.Network import ObjectNet, DNet
from robot.robot import UR5
from rp import setup_basic, dep_dict, scene_graph, dep_graph, remove_objects
from utility import load_model, dist_q, dist_e, jaccard, mean, quat_angle, tid_name, visualize, make_pcd, tid_colors, \
    std


def setup_field_fromdata(node_ids, oid_tid, poss, orns, slow=False):
    """ takes the goal poss and orns and lays out the involved objects in a grid on the plane """
    y_range = [0.2, 0.5]
    x_range = [.2, -.2]
    loader = PBObjectLoader('Generation/urdfc')

    num_objs = len(node_ids)
    g_oids = np.array(sorted(list(oid_tid.keys())))

    idx = 0
    for xpos in np.linspace(*x_range, 5):
        for ypos in np.linspace(*y_range, 5):
            if idx >= num_objs:
                break

            oid = g_oids[idx]
            typ = tid_name(oid_tid[oid])
            pos, orn = poss[idx], orns[idx]

            # modify orn on xy plane
            # rot = R.from_rotvec([0, 0, np.random.uniform(-np.pi/4, np.pi/4)])
            # new_orn = (rot * R.from_quat(orn)).as_quat()
            new_orn = orn

            c_oid = loader.load_obj(otype=typ, pos=(xpos, ypos, 0.01), quat=new_orn, wait=100, slow=slow)
            p.changeDynamics(c_oid, -1, mass=0.05)
            idx += 1

    return loader, g_oids


def get_target_pose(idx, poss, orns):
    return poss[idx], orns[idx]


def fix_object(object_id):
    constraint_id = p.createConstraint(
        parentBodyUniqueId=-1,
        parentLinkIndex=-1,
        childBodyUniqueId=object_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0),
    )
    return constraint_id


def sim_step():
    p.stepSimulation()


def rearrangement_metrics(moved_idx, curr_state, g_poss, g_orns):
    pos_err, orn_err, ora_err = [], [], []
    for obj_idx in moved_idx:
        c_oid = curr_state.obj_ids[obj_idx]

        g_pos, g_orn = g_poss[obj_idx], g_orns[obj_idx]
        c_pos, c_orn = p.getBasePositionAndOrientation(c_oid)

        pos_err.append(dist_e(c_pos, g_pos))
        orn_err.append(dist_q(c_orn, g_orn))

        qa = quat_angle(c_orn, g_orn) % 2 * np.pi
        qa = min(qa, 2 * np.pi - qa)
        ora_err.append(qa)

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
        'pos_err': pos_err, 'pos_mean': np.mean(pos_err), 'pos_std': np.std(pos_err), 'pos_min': np.min(pos_err),
        'pos_max': np.max(pos_err),
        'orn_err': orn_err, 'orn_mean': np.mean(orn_err), 'orn_std': np.std(orn_err), 'orn_min': np.min(orn_err),
        'orn_max': np.max(orn_err),
        # 'ora_err': ora_err, 'ora_mean': np.mean(ora_err), 'ora_std': np.std(ora_err), 'ora_min': np.min(ora_err),
        # 'ora_max': np.max(ora_err),
    }


def planning(node_graph, dep_net, dep_g):
    # infer scene structure/planning
    # start = time.time()
    # pred_graph = scene_graph(node_graph, dep_model=dep_net)  # TODO add adaptation to reduce threshold
    # inference_time = time.time() - start
    # pred_layers = list(toposort(dep_dict(pred_graph)))
    # planning_time = time.time() - start
    gt_graph = dep_g

    # get jaccard similarity of layers # list(toposort(dep_dict(gt_graph)))
    gt_dict = dep_dict(gt_graph)
    # gt_layers = list(toposort(gt_dict))
    # layer_jaccards = [jaccard(gt_l, pr_l) for gt_l, pr_l in zip_longest(list(gt_layers), list(pred_layers),
    #                                                                     fillvalue=set())]
    # mean_jaccard = mean(layer_jaccards)

    return gt_dict, {
        # 'inference_time': inference_time,
        # 'planning_time': planning_time,
        # 'layer_jaccards': layer_jaccards,
        # 'mean_jaccard': mean_jaccard,
        # 'pred_graph': pred_graph,
        # 'gt_graph': gt_graph,
        # 'graphs_equal': (gt_graph == pred_graph).all()
    }


def close(c_pos, c_orn, g_pos, g_orn, p_thres=0.01, o_thres=0.1):
    pos_close = dist_e(c_pos, g_pos) < p_thres
    orn_close = dist_q(c_orn, g_orn, normalized=True) < o_thres
    return pos_close and orn_close


def sim_offset_adjustment(o_id, g_pos, g_orn):
    p.resetBasePositionAndOrientation(o_id, g_pos, g_orn)


def rnext_obj(obj_idxs, gt_dict: dict, already_placed: set):
    tries = 0
    tried = set()
    while True:
        tries += 1
        obj_idx = np.random.choice(list(set(obj_idxs)-already_placed-tried))
        if can_be_placed(obj_idx, gt_dict, already_placed):
            return obj_idx, tries
        tried.add(obj_idx)


def inext_obj(obj_idxs, gt_dict: dict, already_placed: set):
    tries = 0
    for obj_idx in [obj for obj in obj_idxs if obj not in already_placed]:
        tries += 1
        if can_be_placed(obj_idx, gt_dict, already_placed):
            return obj_idx, tries



def can_be_placed(obj_idx: int, gt_dict: dict, already_placed: set):
    return gt_dict[obj_idx] <= already_placed


def rearrangement(gt_dict, curr_state, poss, orns, next_obj, timeout=300000):
    start_time = time.time()

    obj_ids = curr_state.obj_ids
    num_objs = len(obj_ids)
    obj_idxs = np.arange(num_objs)
    moved_idx = []
    move_times = []
    move_try_nums = []

    shuffled_obj_idxs = obj_idxs.copy()
    np.random.shuffle(shuffled_obj_idxs)

    # print(f'NUMBER OF OBJECTS {num_objs}')

    while len(moved_idx) < num_objs:
        start_move = time.time()
        nobj_idx, move_try_num = next_obj(shuffled_obj_idxs, gt_dict, set(moved_idx))
        move_time = time.time() - start_move

        moved_idx.append(nobj_idx)
        move_times.append(move_time)
        move_try_nums.append(move_try_num)

        c_oid = curr_state.obj_ids[nobj_idx]

        # in real, position obtained from point-cloud and orientation from TEASER++
        c_pos_cen, c_orn = p.getBasePositionAndOrientation(c_oid)
        g_pos_cen, g_orn = get_target_pose(nobj_idx, poss, orns)
        c_pos = np.array(c_pos_cen)  # get_suc_point(c_pcds, c_oids, c_oid)
        g_pos_cen = np.array(g_pos_cen)

        # print(f'MOVING {c_oid} to {g_pos_cen}, {g_orn}')
        p.resetBasePositionAndOrientation(c_oid, g_pos_cen, g_orn)
        for _ in range(1000):
            p.stepSimulation()

        if (time.time() - start_time) > timeout:
            return False, moved_idx, rearrangement_metrics(moved_idx, curr_state, poss, orns)

    return True, moved_idx, {**rearrangement_metrics(moved_idx, curr_state, poss, orns),
                             'move_times': np.array(move_times),
                             'move_try_nums': np.array(move_try_nums),
                             'total_move_time': sum(move_times),
                             'total_move_try_num': sum(move_try_nums)
                             }


def setup_env(oid_tid, node_ids, poss, orns, headless=False):
    # add plane and camera and stuff
    setup_basic(headless=headless)

    # set up workspace for rearrangement
    curr_state, g_oids = setup_field_fromdata(node_ids, oid_tid, poss, orns)
    robot = UR5([-0.5, 0, 0])
    for _ in range(100):
        p.stepSimulation()

    return robot, curr_state, g_oids


def get_segment(num, seg_keys):
    for l_bound, u_bound in seg_keys:
        if l_bound < num <= u_bound:
            return l_bound, u_bound
    return None


def main(selection_func, base_type, cur_seg):
    # load models
    feat_net = load_model(ObjectNet, 'cn_test_best_model.pt')
    feat_net.eval()

    dep_net = load_model(DNet, 'dnT_best_model_95_nn.pt',
                         model_args=[511, 256, 128], model_kwargs={'heads': 16, 'concat': False})
    dep_net.eval()

    results = []
    success_p_thresh, success_o_thresh = 0.01, 0.2

    # what scenes to test on
    # l_bound, u_bound = 0, 20
    # data_count = 99999

    # load data and do experiments
    _, _, test_loader = get_scenesdataloader(feat_net)
    print('done loading')
    total_successes = 0
    total_completions = 0

    # data_counts = {(0, 10): 0,
    #                (10, 15): max_examples,
    #                (15, 20): max_examples,
    #                (20, 100): max_examples,
    #                None: max_examples}
    # cur_seg = (0, 10)
    data_counts = {cur_seg: 0}
    max_examples = 1000

    for i, data in enumerate(test_loader):
        # if not condition(data, l_bound, u_bound):
        #     continue
        # if i not in [1134, 472, 490, 523, 570, 858, 1045, 1048, 1134]: # 570 setup issue, 1045 pred issue.
        #     continue
        # count += 1
        # if count >= data_count:
        #     break

        # cur_seg = get_segment(len(data.adj_mat[0]), data_counts.keys())

        lb, ub = cur_seg
        if not (lb < len(data.adj_mat[0]) <= ub):
            continue
        if data_counts[cur_seg] >= max_examples:
            continue

        robot, initial_state, g_oids = setup_env(data.oid_tid[0][0], data.node_ids[0],
                                                 data.g_poss[0], data.g_orns[0], headless=True)

        robot.move_timestep = 0
        gt_dict, p_metrics = planning(data, dep_net, data.adj_mat[0])
        timeout, moved_idx, r_metrics = rearrangement(gt_dict, initial_state, data.g_poss[0], data.g_orns[0],
                                                      selection_func)

        p.disconnect()

        # save/print metrics for scene
        p_s = r_metrics['pos_err'] < success_p_thresh
        o_s = r_metrics['orn_err'] < success_o_thresh
        completion = p_s & o_s

        com, suc = completion.mean(), completion.all()

        if not suc:
            continue

        data_counts[cur_seg] += 1
        cur_count = sum(data_counts.values())

        result = {'moved_idx': np.array(moved_idx),
                  **p_metrics, **r_metrics,
                  'num_nodes': len(data.adj_mat[0]),
                  'completion': com,
                  'success': suc,
                  'data_num': cur_count,
                  'data_idx': i,
                  'seg': cur_seg}
        print(result)
        print(f'[{cur_count}]({i}) success: {suc}, completion: {com}')
        results.append(result)

        # if i % 100 == 0:
        #     seg_str = "-".join(str(x) for x in cur_seg)
        #     file_name = f'results/expert/dp{i}_results_expertrandom_{max_examples}_{seg_str}_{time.time()}'
        #     with open(file_name, 'wb') as f:
        #         pickle.dump(results, f)

        total_completions += com
        total_successes += suc

        print(f'Total success rate: {total_successes / cur_count * 100}')
        print(f'Total completion rate: {total_completions / cur_count * 100}')
        print(f'Number of nodes: {result["num_nodes"]}')
        print(f'Total Move Tries: {result["total_move_try_num"]}, Move Tries: {result["move_try_nums"]}')
        print(f'Total Move time: {result["total_move_time"]}\n\n')

        print(data_counts)

    # save/rpint all metrics
    success_rate = mean([r['success'] for r in results])
    completion_rate = mean([r['completion'] for r in results])

    avg_pos_mean = mean([r['pos_mean'] for r in results])
    avg_pos_std = mean([r['pos_std'] for r in results])
    avg_orn_mean = mean([r['orn_mean'] for r in results])
    avg_orn_std = mean([r['orn_std'] for r in results])

    # avg_pred_time = mean([r['inference_time'] for r in results])
    # avg_plan_time = mean([r['planning_time'] for r in results])
    # avg_mean_jacc = mean([r['mean_jaccard'] for r in results])
    #
    # std_pred_time = std([r['inference_time'] for r in results])
    # std_plan_time = std([r['planning_time'] for r in results])
    # std_mean_jacc = std([r['mean_jaccard'] for r in results])

    print('\n\n[FINAL METRICS]')
    print(f'success rate: {success_rate*100}, completion rate: {completion_rate*100}\n')

    print(f'mean position error:\t\t{avg_pos_mean}+/-{avg_pos_std} (m)')
    print(f'mean orientation error:\t\t{avg_orn_mean}+/-{avg_orn_std} (0-1)')
    # print(f'structure inference time:\t{avg_pred_time}+/-{std_pred_time} (s)')
    # print(f'mean planning time\t\t\t{avg_plan_time}+/-{std_plan_time} (s)')
    # print(f'averaged mean jaccard\t\t{avg_mean_jacc}+/-{std_mean_jacc}')

    seg_str = "-".join(str(x) for x in cur_seg)
    file_name = f'results/classical/results_{base_type}_{max_examples}_{seg_str}_{int(time.time())}'
    with open(file_name, 'wb') as f:
        pickle.dump(results, f)

    # recreate_scene(9000)
    # time.sleep(10000)


if __name__ == '__main__':
    for cur_seg in [(10, 15), (15, 20), (20, 25), (25, 50)]:
        for sel_func, b_type in zip([inext_obj, rnext_obj], ['iterative', 'random']):
            main(sel_func, b_type, cur_seg)
