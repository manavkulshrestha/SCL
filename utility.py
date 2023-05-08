import numpy as np
from scipy.sparse import coo_matrix

from os import path as osp

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from Dataset import DependenceDataset, ObjectDataset

import open3d as o3d


device = torch.device('cuda')
RP_ROOT = osp.abspath('/home/mk/rp/')
DDPATH = osp.join(RP_ROOT, 'data/dep_data/')
PDPATH = osp.join(RP_ROOT, 'data/pcd_data/')
MODELS_PATH = osp.join(RP_ROOT, 'models/')


def sliding(lst, n):
    for window in zip(*[lst[i:] for i in range(n)]):
        yield window


def coo_withzeros(it, num_nodes, rowcol, adj_graph):
    depg = np.zeros((num_nodes, num_nodes))
    depg[tuple(rowcol)] = 1

    ones_loc = np.array(np.where(depg == 1))
    zeros_loc = np.array(np.where(depg == 0))

    edge_label_index = np.hstack([ones_loc, zeros_loc])
    edge_label = np.hstack([np.ones(ones_loc.shape[1]), np.zeros(zeros_loc.shape[1])])

    return edge_label_index, edge_label


def get_rowcol(adj_graph):
    adj = coo_matrix(adj_graph)
    return np.vstack([adj.row, adj.col])


def load_depgs(data_dir):
    ret = []

    for i in range(10000):
        npzfile = np.load(osp.join(data_dir, f'{i // 1000}_{i % 1000}.npz'))
        depg = npzfile['depg']

        ret.append((depg, get_rowcol(depg), len(depg)))

    return ret


def get_depdataloaders(feat_net):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
    ])

    sc = 512
    train_dataset = DependenceDataset(PDPATH, DDPATH, feat_net=feat_net, chunk=(0, 8000), transform=transform, sample_count=sc)
    val_dataset = DependenceDataset(PDPATH, DDPATH, feat_net=feat_net, chunk=(8000, 9000), transform=transform, sample_count=sc)
    test_dataset = DependenceDataset(PDPATH, DDPATH, feat_net=feat_net, chunk=(9000, 10000), transform=transform, sample_count=sc)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # num workers causes error
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)  # num workers causes error
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # num workers causes error

    return train_loader, val_loader, test_loader


def get_objdataloaders():
    # transform = T.Compose([
    #     T.NormalizeFeatures(),
    #     T.ToDevice(device),
    # ])

    # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    pre_transform, transform = T.NormalizeScale(), None

    sc = 512
    train_dataset = ObjectDataset(PDPATH, chunk=(0, 500), pre_transform=pre_transform, transform=transform, sample_count=sc)
    val_dataset = ObjectDataset(PDPATH, chunk=(500, 550), pre_transform=pre_transform, transform=transform, sample_count=sc)
    test_dataset = ObjectDataset(PDPATH, chunk=(550, 600), pre_transform=pre_transform, transform=transform, sample_count=sc)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # num workers causes error
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)  # num workers causes error
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # num workers causes error

    return train_loader, val_loader, test_loader


def save_model(model, name):
    torch.save(model.state_dict(), osp.join(MODELS_PATH, name))


def tid_colors(typeidx):
    return np.array([
        [0, 0, 0],
        [0.4160655362053999, 0.13895618220307226, 0.05400398384796701],
        [0.45815538934366873, 0.5622777225161942, 0.12222557471515583],
        [0.5285494304846649, 0.8052616853729326, 0.47328724440755865],
        [0.520059254934221, 0.4733167572634138, 0.5049641813650521],
        [0.2448837131753191, 0.5174992612426157, 0.8959927219176224],
        [0.0859375, 0.9921875, 1.3359375],
        [0.8728815094572616, 0.11715588167789504, 0.9012921785976408],
        [0.8708585184367256, 0.13537291463132384, 0.2942509320637464]
    ])[typeidx]


def make_pcd(xyz, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        if len(colors) == 3:
            colors = np.tile(colors, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize(pcd):
    o3d.visualization.draw_geometries([pcd])
    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window()
    #
    # viewer.add_geometry(pcd_yescol)
    # box = o3d.geometry.AxisAlignedBoundingBox([-1, -1, 0], [1, 1, 1])
    # box.color = [1, 0, 0]  # red color
    # # box.alpha = 0.1  # set alpha value to 0.1
    # viewer.add_geometry(box)
    #
    # opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    # viewer.run()
    # viewer.destroy_window()


def sample_exact(pcd, n):
    idx = np.random.choice(len(pcd), n, replace=len(pcd) < n)
    return pcd[idx]


def normalize(inp):
    # Center the input
    center = np.mean(inp, axis=0)
    centered_inp = inp - center

    # Scale the input to (-1, 1)
    scale = (1 / np.abs(centered_inp).max()) * 0.999999
    normalized_inp = centered_inp * scale

    return normalized_inp

# def normalize(inp):
#     # Center the input
#     center = torch.mean(inp, dim=0)
#     centered_inp = inp - center.unsqueeze(0)
#
#     # Scale the input to (-1, 1)
#     scale = (1 / torch.abs(centered_inp).max()) * 0.999999
#     normalized_inp = centered_inp * scale
#
#     return normalized_inp
