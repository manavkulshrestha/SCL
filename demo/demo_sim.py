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

from Datasets.dutility import PDPATH, DDPATH, get_depdataloaders
from Generation.gen_lib import simulate_scene_pc, Camera, PBObjectLoader
from nn.Network import ObjectNet, DNet
from nn.PositionalEncoder import PositionalEncoding
from robot.robot import UR5
from testing.tutility import print_dep, plot_adj_mats
from utility import load_model, all_edges, name_tid, tid_name, map_dict, draw_sphere_marker, dist_q, dist_e, jaccard, \
    mean, quat_angle


debug = False


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


def setup_field(loader_target, slow=False):
    """ takes the goal state and lays out the involved objects in a grid on the plane """
    loader2 = PBObjectLoader('Generation/urdfc')

    y_range = [-0.4, -0.2]
    x_range = [-.2, .2]

    num_objs = len(loader_target.obj_poses)

    idx = 0
    for xpos in np.linspace(*x_range, 4):
        for ypos in np.linspace(*y_range, 3):
            if idx >= num_objs:
                break

            oid = loader_target.obj_ids[idx]
            typ = loader_target.oid_typ_map[oid]
            pos, orn = loader_target.obj_poses[oid]

            # modify orn on xy plane
            # rot = R.from_rotvec([0, 0, np.random.uniform(-np.pi/2, np.pi/2)])
            # new_orn = (rot * R.from_quat(orn)).as_quat()
            # new_orn = orn
            new_orn = (0, 0, -np.pi/2) if typ == 'ccuboid' else (0, 0, 0)

            c_oid = loader2.load_obj(otype=typ, pos=(xpos, ypos, 0.01), euler=new_orn, wait=100, slow=slow)
            p.changeDynamics(c_oid, -1, mass=0.01)
            idx += 1

    return loader2


def remove_objects(loader):
    """ removes all objects in the state from the scene """
    for oid in loader.obj_ids:
        p.removeBody(oid)


def get_obj_feats(obj_pcd, sample_count=512, *, feat_model, pos_enc):
    """ returns the features for an object from its point cloud """
    obj_cen = obj_pcd.mean(axis=0)

    if sample_count is not None:
        idx = np.random.choice(len(obj_pcd), sample_count, replace=len(obj_pcd) < sample_count)
        obj_pcd = obj_pcd[idx]

    # get total features from positional encoding of centroid and object level features from network
    cen_ten = torch.tensor(obj_cen, dtype=torch.float).cuda()
    pred_tid, obj_emb = feat_model.embed(obj_pcd, get_pred=True)

    obj_emb = torch.squeeze(obj_emb)
    pos_emb = pos_enc(cen_ten)
    x = torch.cat([pos_emb, obj_emb])

    return x, pred_tid


def dependence(obj1, obj2):
    """
    returns {
         0 if objects are independent
         1 if object 1 depends on object 2
    }
    """
    ctpts = p.getContactPoints(int(obj1), int(obj2))
    if len(ctpts) == 0:
        return 0
    contact_vec_on_2 = ctpts[0][7]

    return 1 if np.sign(contact_vec_on_2)[-1] == 1 else 0


def dep_graph(node_ids):
    """
    returns an adjacency matrix with shape (len(node_ids),len(node_ids)) where {
        1 == depg[i,j] means i depends on j
        0 == depg[i,j] means i does not depend on j
    }

    note: lower/higher refers to the index in the matrix, not the object ids
    """
    n = len(node_ids)
    depg = np.zeros((n, n))

    for i, obj1 in enumerate(node_ids):
        for j, obj2 in enumerate(node_ids):
            depg[i, j] = dependence(obj1, obj2)

    return depg


def initial_graph(pcds, oids, *, feat_model):
    """ takes scene point cloud and returns an initial scene graph """
    pos_enc = PositionalEncoding(min_deg=0, max_deg=5, scale=1, offset=0).cuda()

    nodes_feats = []
    for oid in np.unique(oids):  # in real, oid is gotten by instance segmentation
        pcd = pcds[oid == oids]
        node_feats, pred_tid = get_obj_feats(pcd, feat_model=feat_model, pos_enc=pos_enc)

        nodes_feats.append(node_feats)

    return Data(x=torch.stack(nodes_feats).cpu(), all_e_idx=all_edges(len(nodes_feats)), num_nodes=len(nodes_feats))


def scene_graph(node_graph, *, dep_model, thresh=0.5):
    node_graph = node_graph.cuda()
    threshold = torch.tensor(thresh).cuda()

    out = dep_model(node_graph.x, node_graph.all_e_idx)
    outs = out.sigmoid()
    pred = (outs > threshold).float()

    pred_adj = np.zeros((node_graph.num_nodes, node_graph.num_nodes))
    pred_e_idx = node_graph.all_e_idx[:, pred.bool()].cpu().numpy()
    pred_adj[tuple(pred_e_idx)] = 1

    return pred_adj


def load_scene(scene_num):
    scene_name = f'{scene_num // 1000}_{scene_num % 1000}.npz'
    pcd_path = osp.join(PDPATH, scene_name)
    dep_path = osp.join(DDPATH, scene_name)

    pcd_file = np.load(pcd_path)
    dep_file = np.load(dep_path)
    pcds, tids, oids = [pcd_file[x] for x in ['pc', 'tid', 'oid']]
    node_ids, dep_g = [dep_file[x] for x in ['node_ids', 'depg']]

    return pcds, oids


def dep_dict(adj_mat):
    """ creates a dictionary representation of the graph """
    return {i: set(np.nonzero(row)[0]) for i, row in enumerate(adj_mat)}


def get_target_pose(loader_t, oid):
    """ get position and orientation, in real this is from teaser++ and point cloud centroid """
    return loader_t.obj_poses[oid]


def get_suc_point(pcds, oids, oid, epsilon=0.00001):
    o_pcd = pcds[oids == oid]

    max_z = np.max(o_pcd[:, 2])
    uppermost_idx = (o_pcd[:, 2] >= max_z - epsilon) & (o_pcd[:, 2] <= max_z + epsilon)

    uppermost_pts = o_pcd[uppermost_idx]
    return uppermost_pts.mean(axis=0)


def main():
    # seed = 1369 or np.random.randint(0, 10000)
    # seed = 500 or np.random.randint(0, 10000)
    # seed = 9457 or np.random.randint(0, 10000)
    # seed = 8634 or np.random.randint(0, 10000)
    # seed = 3097 or np.random.randint(0, 10000)
    # seed = 4276 or np.random.randint(0, 10000)
    # seed = 8174 or np.random.randint(0, 10000)
    # seed = 4978 or np.random.randint(0, 10000) # pred error, but recovers

    # seed = 4978

    seed = np.random.randint(0, 10000)
    print(f'SEED: {seed}')
    np.random.seed(seed)

    # load models
    feat_net = load_model(ObjectNet, 'cn_test_best_model.pt')
    feat_net.eval()

    dep_net = load_model(DNet, 'dnT_best_model_95_nn.pt',
                         model_args=[511, 256, 128], model_kwargs={'heads': 16, 'concat': False})
    dep_net.eval()

    # setup the environment
    setup_basic()
    cams = [Camera(pos) for pos in [[0.15, 0.15, .2], [0.15, -0.15, .2], [0, 0, .3]]]

    # set up the target scene
    goal_state, pcds, oids = simulate_scene_pc(cams, slow=False, wait=100)

    # infer scene structure
    node_graph = initial_graph(pcds, oids, feat_model=feat_net)
    start = time.time()
    pred_graph = scene_graph(node_graph, dep_model=dep_net)  # TODO add adaptation to reduce threshold
    planning_time = time.time() - start
    pred_layers = toposort(dep_dict(pred_graph))
    withsorting_time = time.time() - start
    gt_graph = dep_graph(goal_state.obj_ids)

    # get jaccard similarity of layers
    gt_layers = toposort(dep_dict(gt_graph))
    jacc = [jaccard(gt_l, pr_l) for gt_l, pr_l in zip_longest(gt_layers, pred_layers, fillvalue=set())]
    javg = mean(jacc)

    # plot_adj_mats(gt_graph, pred_graph, titles=['Ground Truth', 'Prediction'])
    # plt.show()

    # if (gt_graph != pred_graph).any():
    # assert (gt_graph == pred_graph).all()

    # set up workspace for rearrangement
    time.sleep(1)
    remove_objects(goal_state)
    curr_state = setup_field(goal_state)
    cam4_pos = [-0.25, 0, 0.5]
    # cam4 = Camera(cam4_pos, target=[*cam4_pos[:2], 0])
    # c_pcds, c_oids = cam4.get_point_cloud()
    robot = UR5([-0.5, 0, 0])
    for _ in range(100):
        p.stepSimulation()

    # planning
    graph_dict = dep_dict(pred_graph)
    topo_layers = toposort(graph_dict)
    robot.move_timestep = 1/240
    robot.move_timestep = 0

    # while True:
    #     p.stepSimulation()
    #     time.sleep(1/240)

    # rearrangement
    moved_idx = []
    for l_num, layer in enumerate(topo_layers):
        for obj_idx in layer:
            moved_idx.append(obj_idx)
            c_oid = curr_state.obj_ids[obj_idx]
            g_oid = goal_state.obj_ids[obj_idx]

            if c_oid == 1:
                # robot.move_timestep = 1/60
                print(1)

            # in real, position obtained from point-cloud and orientation from TEASER++
            c_pos_cen, c_orn = p.getBasePositionAndOrientation(c_oid)
            g_pos_cen, g_orn = get_target_pose(goal_state, g_oid)
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
            rotated_succ_offt = g_orn_mat@succ_offt
            g_pos_to = g_pos_cen+rotated_succ_offt

            # move above goal position, move to goal, drop, move above goal
            robot.move_ee_above(g_pos_to, orn=g_orn_to)
            robot.move_ee(g_pos_to+[0, 0, 0.002], orn=g_orn_to)

            for _ in range(100): # block has inertia from the robot moving
                p.stepSimulation()
                if debug:
                    time.sleep(robot.move_timestep)

            robot.suction(False)

            # p.changeDynamics(c_oid, -1, mass=0.1)
            for _ in range(500):
                p.stepSimulation()
                if debug:
                    time.sleep(robot.move_timestep)

            robot.move_ee_above(g_pos_cen, orn=(0, 0, 0, 1))
            p.changeDynamics(c_oid, -1, mass=[0.5, 0.02, 0.01][l_num])
            # p.resetBasePositionAndOrientation(c_oid, g_pos_cen, g_orn)  # TODO for testing, remove later


    # metric collection
    pos_err, orn_err, ora_err = [], [], []
    for obj_idx in moved_idx:
        g_oid = goal_state.obj_ids[obj_idx]
        c_oid = curr_state.obj_ids[obj_idx]

        g_pos, g_orn = goal_state.obj_poses[g_oid]
        c_pos, c_orn = p.getBasePositionAndOrientation(c_oid)

        pos_err.append(dist_e(c_pos, g_pos))
        orn_err.append(dist_q(c_orn, g_orn))
        ora_err.append(quat_angle(c_orn, g_orn))

    # report metrics
    pos_err = np.array(pos_err)
    orn_err = np.array(orn_err)
    print(pos_err)
    print(orn_err)
    orn_err /= np.sqrt(2)

    print(f'pos error: {np.mean(pos_err):.4f}+/-{np.std(pos_err):.4f}, max: {np.max(pos_err):.4f}, min: {np.min(pos_err):.4f}')
    print(f'orn error: {np.mean(orn_err):.4f}+/-{np.std(orn_err):.4f}, max: {np.max(orn_err):.4f}, min: {np.min(orn_err):.4f}')
    print(f'ora error: {np.mean(ora_err):.4f}+/-{np.std(ora_err):.4f}, max: {np.max(ora_err):.4f}, min: {np.min(ora_err):.4f}')
    print(f'averaged jaccard similarity of inferred layers: {javg}')
    print(f'planning time: {planning_time:.6f} (dependence graph) and {withsorting_time:.6f} (with sorting)')

    time.sleep(100000)

    # logging in file, analysis/readout script


if __name__ == '__main__':
    main()
