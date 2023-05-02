import numpy as np
from scipy.sparse import coo_matrix

from os import path as osp

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from Dataset import DependenceDataset

device = torch.device('cuda')
RP_ROOT = osp.abspath('../../rp/')
DDPATH = osp.join(RP_ROOT, 'data/dep_data/')
MODEL_PATH = osp.join(RP_ROOT, 'models/')


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


def get_dataloaders():
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
    ])

    train_dataset = DependenceDataset(DDPATH, chunk=(0, 8000), transform=transform)
    val_dataset = DependenceDataset(DDPATH, chunk=(8000, 9000), transform=transform)
    test_dataset = DependenceDataset(DDPATH, chunk=(9000, 10000), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # num workers causes error
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)  # num workers causes error
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # num workers causes error

    return train_loader, val_loader, test_loader


def save_model(model, name):
    torch.save(model.state_dict(), osp.join(MODEL_PATH, name))