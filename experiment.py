import pickle

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

        qa = quat_angle(c_orn, g_orn) % 2*np.pi
        qa = min(qa, 2*np.pi - qa)
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
        'pos_err': pos_err, 'pos_mean': np.mean(pos_err), 'pos_std': np.std(pos_err), 'pos_min': np.min(pos_err), 'pos_max': np.max(pos_err),
        'orn_err': orn_err, 'orn_mean': np.mean(orn_err), 'orn_std': np.std(orn_err), 'orn_min': np.min(orn_err), 'orn_max': np.max(orn_err),
        'ora_err': ora_err, 'ora_mean': np.mean(ora_err), 'ora_std': np.std(ora_err), 'ora_min': np.min(ora_err), 'ora_max': np.max(ora_err),
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
        'mean_jaccard': mean_jaccard,
        'pred_graph': pred_graph,
        'gt_graph': gt_graph,
        'graphs_equal': (gt_graph == pred_graph).all()
    }


def close(c_pos, c_orn, g_pos, g_orn, p_thres=0.01, o_thres=0.1):
    pos_close = dist_e(c_pos, g_pos) < p_thres
    orn_close = dist_q(c_orn, g_orn, normalized=True) < o_thres
    return pos_close and orn_close


def sim_offset_adjustment(o_id, g_pos, g_orn):
    p.resetBasePositionAndOrientation(o_id, g_pos, g_orn)


def rearrangement(robot, pred_layers, curr_state, poss, orns, timeout=300000, control_fix=False):
    start_time = time.time()

    moved_idx = []
    for l_num, layer in enumerate(pred_layers):
        for obj_idx in layer:
            moved_idx.append(obj_idx)
            c_oid = curr_state.obj_ids[obj_idx]

            # in real, position obtained from point-cloud and orientation from TEASER++
            c_pos_cen, c_orn = p.getBasePositionAndOrientation(c_oid)
            g_pos_cen, g_orn = get_target_pose(obj_idx, poss, orns)
            c_pos = np.array(c_pos_cen)  # get_suc_point(c_pcds, c_oids, c_oid)
            g_pos_cen = np.array(g_pos_cen)

            # obtain goal orientation
            g_orn_to = p.getDifferenceQuaternion(c_orn, g_orn)  # in real, done with TEASER++

            # move above cur position, move to curr, pick, move above curr
            robot.move_ee_above(c_pos, orn=(0, 0, 0, 1), above_offt=(0, 0, 0.2))
            robot.move_ee_above(c_pos, orn=(0, 0, 0, 1), above_offt=(0, 0, 0.05))
            c_pos_from, _ = robot.move_ee_down(c_pos, orn=(0, 0, 0, 1))
            robot.suction(True)
            robot.move_ee_above(c_pos, orn=(0, 0, 0, 1))

            # obtain goal pose
            succ_offt = np.subtract(c_pos_from, c_pos_cen)
            g_orn_mat = R.from_quat(g_orn_to).as_matrix()
            rotated_succ_offt = g_orn_mat @ succ_offt
            g_pos_to = g_pos_cen + rotated_succ_offt

            # move above goal position, move to goal, drop, move above goal
            robot.move_ee_above(g_pos_to, orn=g_orn_to)
            robot.move_ee(g_pos_to + [0, 0, 0.003], orn=g_orn_to)

            for _ in range(100):  # block has inertia from the robot moving
                p.stepSimulation()

            robot.suction(False)
            p.changeDynamics(c_oid, -1, mass=0.00001)

            if control_fix and close(*p.getBasePositionAndOrientation(c_oid), g_pos_cen, g_orn):
                sim_offset_adjustment(c_oid, g_pos_cen, g_orn)

            for _ in range(500):
                p.stepSimulation()

            robot.move_ee_above(g_pos_cen, orn=(0, 0, 0, 1))
            xl_num = min(2, l_num)
            p.changeDynamics(c_oid, -1, mass=[0.5, 0.02, 0.01][xl_num])
            # fix_object(c_oid)
            if (time.time() - start_time) > timeout:
                return False, moved_idx, rearrangement_metrics(moved_idx, curr_state, poss, orns)

    return True, moved_idx, rearrangement_metrics(moved_idx, curr_state, poss, orns)


def setup_env(oid_tid, node_ids, poss, orns, headless=False):
    # add plane and camera and stuff
    setup_basic(headless=headless)

    # set up workspace for rearrangement
    curr_state, g_oids = setup_field_fromdata(node_ids, oid_tid, poss, orns)
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

    results = []
    success_p_thresh, success_o_thresh = 0.01, 0.2

    # what scenes to test on
    l_bound, u_bound = 0, 10
    data_count = 100
    valid_is = {5, 18, 22, 35, 47, 71, 80, 112, 116, 123, 130, 147, 166, 194, 223, 237, 249, 259, 290, 299, 301, 314, 340, 347, 358, 365, 371, 386, 389, 391, 405, 409, 410, 425, 427, 443, 472, 510, 523, 543, 550, 551, 570, 579, 586, 615, 663, 670, 713, 715, 717, 733, 744, 757, 761, 769, 775, 806, 808, 809, 858, 884, 930, 932, 939, 944, 952, 967, 994, 995, 1002, 1003, 1004, 1011, 1015, 1016, 1021, 1030, 1039, 1041, 1045, 1048, 1060, 1064, 1071, 1089, 1091, 1112, 1115, 1124, 1127, 1134, 1149, 1156, 1160, 1175, 1179, 1181, 1184, 1204, 1210, 1213, 1217, 1223, 1231, 1242, 1300, 1333, 1380, 1381, 1383, 1384, 1401, 1440, 1442, 1453, 1454, 1461, 1465, 1466, 1472, 1525, 1555, 1561, 1567, 1577, 1590, 1592, 1594, 1600, 1608, 1627, 1655, 1690, 1702, 1711, 1715, 1722, 1726, 1753, 1758, 1763, 1791, 1797, 1816, 1819, 1846, 1854, 1861, 1880, 1892, 1895, 1903, 1904, 1906, 1907, 1911, 1975, 1991, 1995, 2008, 2011, 2014, 2023, 2040, 2113, 2144, 2152, 2162, 2170, 2173, 2187, 2194, 2199, 2229, 2243, 2244, 2246, 2257, 2259, 2263, 2272, 2276, 2318, 2330, 2334, 2336, 2343, 2353, 2368, 2374, 2400, 2413, 2439, 2445, 2448, 2468, 2472, 2487, 2523, 2536, 2538, 2548, 2549, 2550, 2594, 2602, 2643, 2646, 2659, 2667, 2678, 2679, 2680, 2688, 2700, 2701, 2721, 2734, 2764, 2777, 2790, 2808, 2856, 2868, 2869, 2878, 2895, 2905, 2908, 2910, 2919, 2959, 2972, 2990, 3010, 3043, 3049, 3057, 3077, 3103, 3109, 3157, 3175, 3190, 3207, 3210, 3238, 3254, 3265, 3281, 3282, 3304, 3314, 3320, 3326, 3353, 3368, 3370, 3378, 3390, 3391, 3400, 3407, 3420, 3426, 3441, 3449, 3458, 3472, 3483, 3490, 3499, 3523, 3540, 3550, 3555, 3561, 3569, 3576, 3598, 3600, 3605, 3607, 3628, 3645, 3668, 3671, 3676, 3687, 3706, 3707, 3710, 3712, 3721, 3742, 3743, 3760, 3766, 3817, 3822, 3828, 3831, 3851, 3876, 3898, 3920, 3921, 3932, 3963, 3964, 3983, 3985, 3987, 3988, 4031, 4049, 4053, 4057, 4065, 4071, 4114, 4137, 4158, 4164, 4206, 4212, 4220, 4234, 4241, 4251, 4277, 4379, 4391, 4400, 4442, 4479, 4489, 4505, 4520, 4524, 4543, 4546, 4555, 4563, 4572, 4587, 4622, 4630, 4636, 4648, 4665, 4671, 4679, 4680, 4742, 4771, 4800, 4827, 4831, 4851, 4861, 4884, 4887, 4925, 4950, 4955, 4976, 4980, 4984, 4996, 5022, 5080, 5088, 5124, 5162, 5164, 5165, 5170, 5184, 5200, 5211, 5214, 5223, 5244, 5259, 5261, 5276, 5277, 5284, 5291, 5347, 5360, 5369, 5373, 5374, 5381, 5416, 5422, 5450, 5461, 5471, 5492, 5497, 5498, 5519, 5520, 5521, 5526, 5539, 5555, 5570, 5571, 5604, 5624, 5644, 5654, 5689, 5697, 5716, 5725, 5763, 5767, 5771, 5777, 5793, 5811, 5849, 5850, 5855, 5864, 5888, 5891, 5894, 5924, 5926, 5934, 5957, 5960, 5963, 5984, 6010, 6026, 6027, 6050, 6055, 6091, 6095, 6112, 6116, 6124, 6130, 6132, 6135, 6142, 6151, 6153, 6157, 6181, 6196, 6216, 6234, 6248, 6269, 6271, 6278, 6287, 6293, 6294, 6312, 6330, 6337, 6348, 6366, 6383, 6392, 6396, 6421, 6422, 6453, 6459, 6486, 6497, 6508, 6510, 6516, 6524, 6539, 6543, 6544, 6557, 6558, 6565, 6611, 6641, 6671, 6685, 6699, 6736, 6756, 6779, 6836, 6837, 6845, 6901, 6904, 6924, 6947, 6949, 6950, 6953, 6963, 7008, 7035, 7047, 7051, 7062, 7067, 7103, 7105, 7126, 7147, 7162, 7163, 7164, 7170, 7173, 7180, 7185, 7189, 7209, 7212, 7223, 7230, 7243, 7244, 7248, 7274, 7285, 7331, 7353, 7362, 7368, 7370, 7377, 7389, 7395, 7411, 7460, 7479, 7488, 7518, 7524, 7530, 7570, 7586, 7589, 7591, 7594, 7607, 7618, 7629, 7641, 7666, 7681, 7715, 7723, 7743, 7748, 7757, 7772, 7781, 7796, 7806, 7835, 7847, 7853, 7863, 7866, 7882, 7908, 7911, 7912, 7925, 7938, 7943, 7962, 7973, 7977, 7984, 7992, 8002, 8018, 8022, 8037, 8079, 8090, 8114, 8149, 8164, 8181, 8216, 8226, 8236, 8260, 8262, 8266, 8291, 8294, 8303, 8334, 8340, 8354, 8362, 8385, 8431, 8435, 8451, 8458, 8460, 8468, 8473, 8482, 8494, 8504, 8510, 8525, 8542, 8552, 8554, 8562, 8567, 8579, 8589, 8592, 8601, 8608, 8622, 8636, 8637, 8648, 8651, 8667, 8687, 8716, 8728, 8734, 8744, 8758, 8765, 8767, 8788, 8793, 8802, 8814, 8837, 8850, 8851, 8873, 8874, 8919, 8928, 8934, 8940, 8944, 8957, 8963, 8970, 9010, 9026, 9033, 9043, 9064, 9118, 9135, 9141, 9151, 9177, 9179, 9191, 9193, 9196, 9237, 9276, 9279, 9283, 9307, 9319, 9361, 9374, 9396, 9424, 9443, 9462, 9464, 9479, 9496, 9499, 9501, 9511, 9516, 9518, 9529, 9560, 9586, 9587, 9597, 9618, 9621, 9633, 9681, 9686, 9691, 9728, 9762, 9776, 9787, 9811, 9820, 9830, 9834, 9852, 9868, 9895, 9906, 9917, 9924, 9941, 9954, 9956}

    # load data and do experiments
    _, _, test_loader = get_scenesdataloader(feat_net)
    count = 0
    print('done loading')
    total_successes = 0
    total_completions = 0
    for i, data in enumerate(test_loader):
        if not condition(data, l_bound, u_bound):
            continue
        # if i not in [1134, 472, 490, 523, 570, 858, 1045, 1048, 1134]: # 570 setup issue, 1045 pred issue.
        #     continue
        if i not in valid_is:
            continue
        count += 1
        if count >= data_count:
            break

        try:
            robot, initial_state, g_oids = setup_env(data.oid_tid[0][0], data.node_ids[0],
                                                     data.g_poss[0], data.g_orns[0], headless=True)

            robot.move_timestep = 0
            pred_layers, p_metrics = planning(data, dep_net, data.adj_mat[0])
            timeout, moved_idx, r_metrics = rearrangement(robot, pred_layers, initial_state, data.g_poss[0], data.g_orns[0],
                                                          control_fix=True)

            p.disconnect()
        except TypeError as e:
            count -= 1
            continue

        # save/print metrics for scene
        p_s = r_metrics['pos_err'] < success_p_thresh
        o_s = r_metrics['orn_err'] < success_o_thresh
        completion = p_s & o_s

        com, suc = completion.mean(), completion.all()

        result = {'moved_idx': moved_idx,
                  **p_metrics, **r_metrics,
                  'num_nodes': len(data.adj_mat[0]),
                  'completion': com,
                  'success': suc,
                  'data_num': count,
                  'data_idx': i}
        print(result)
        print(f'[{count}]({i}) success: {suc}, completion: {com}')
        results.append(result)

        total_completions += com
        total_successes += suc

        print(f'Total success rate: {total_successes/count * 100}')
        print(f'Total completion rate: {total_completions/count * 100}')

    # save/rpint all metrics
    success_rate = mean([r['success'] for r in results])
    completion_rate = mean([r['completion'] for r in results])

    avg_pos_mean = mean([r['pos_mean'] for r in results])
    avg_pos_std = mean([r['pos_std'] for r in results])
    avg_orn_mean = mean([r['orn_mean'] for r in results])
    avg_orn_std = mean([r['orn_std'] for r in results])

    avg_pred_time = mean([r['inference_time'] for r in results])
    avg_plan_time = mean([r['planning_time'] for r in results])
    avg_mean_jacc = mean([r['mean_jaccard'] for r in results])

    std_pred_time = std([r['inference_time'] for r in results])
    std_plan_time = std([r['planning_time'] for r in results])
    std_mean_jacc = std([r['mean_jaccard'] for r in results])

    print('\n\n[FINAL METRICS]')
    print(f'success rate: {success_rate}, completion rate: {completion_rate}\n')

    print(f'mean position error:\t\t{avg_pos_mean}+/-{avg_pos_std} (m)')
    print(f'mean orientation error:\t\t{avg_orn_mean}+/-{avg_orn_std} (0-1)')
    print(f'structure inference time:\t{avg_pred_time}+/-{std_pred_time} (s)')
    print(f'mean planning time\t\t\t{avg_plan_time}+/-{std_plan_time} (s)')
    print(f'averaged mean jaccard\t\t{avg_mean_jacc}+/-{std_mean_jacc}')

    with open(f'results/results_{count}_{l_bound}-{u_bound}_{time.time()}', 'wb') as f:
        pickle.dump(results, f)

    # recreate_scene(9000)
    # time.sleep(10000)


if __name__ == '__main__':
    main()
