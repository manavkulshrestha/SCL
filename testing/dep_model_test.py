import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt

from Datasets.dutility import get_depdataloaders, PDPATH
from nn.Network import DNet, ObjectNet
from utility import load_model, tid_colors, make_pcd, visualize, tid_name

import os.path as osp
import threading


def visualize_scene(scene_num):
    scene_name = f'{scene_num // 1000}_{scene_num % 1000}.npz'
    scene_path = osp.join(PDPATH, scene_name)

    pcd_file = np.load(scene_path)
    pcds, tids, oid = [pcd_file[x] for x in ['pc', 'tid', 'oid']]

    o3dpcds = make_pcd(pcds, colors=tid_colors(tids))
    visualize(o3dpcds)


def scene_graph_info(scene_num):
    scene_name = f'{scene_num // 1000}_{scene_num % 1000}.npz'
    pcd_path = osp.join(PDPATH, scene_name)

    pcd_file = np.load(pcd_path)
    pcds, tids, oids = [pcd_file[x] for x in ['pc', 'tid', 'oid']]

    colors = []
    labels = {}
    for oid in np.unique(oids):
        tid = tids[oid == oids][0]

        oid_m = int(oid)-1

        tc = tid_colors(tid)
        colors.append(np.clip(tc, 0, 1))
        # labels[oid_m] = f'{tid_name(tid)}[{oid_m}]'
        labels[oid_m] = f'{oid_m}'

    return colors, labels


def visualize_graph(adj, colors, labels, ax):
    G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
    nx.draw(G, with_labels=True, labels=labels, node_color=colors, node_size=100, font_size=12, ax=ax)


def plot_adj_mats(*adjs, titles=None):
    fig, axes = plt.subplots(1, 2)
    names = titles if titles is not None else ['']*len(adjs)
    ticks = [f'o-{i}' for i in range(1, len(adjs[0])+1)]

    imgs = [None]*len(adjs)
    for i, (adj, title) in enumerate(zip(adjs, titles)):
        axes[i].imshow(adj, interpolation='nearest')
        axes[i].title.set_text(names[i])
        # axes[i].set_xticklabels(ticks)
        # axes[i].set_yticklabels(ticks)

    fig.suptitle("Adjacency Matrix Prediction")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.1, 0.15, 0.05, 0.7])
    fig.colorbar(imgs[-1], cax=cbar_ax)
    fig.tight_layout()


@torch.no_grad()
def vis_test(model, loader, thresh=0.5):
    model.eval()

    for i, data in enumerate(loader, start=9000):
        out = model(data.x, data.all_e_idx)
        outs = out.sigmoid()
        pred = (outs > thresh).float()

        pred_adj = np.zeros((data.num_nodes, data.num_nodes))
        pred_e_idx = data.all_e_idx[:, pred.bool()].cpu().numpy()
        pred_adj[tuple(pred_e_idx)] = 1

        def adj_vis(i, pred_adj, adj_mat):
            fig, axes = plt.subplots(1, 2)
            visualize_graph(pred_adj, *scene_graph_info(i), ax=axes[0])
            visualize_graph(adj_mat, *scene_graph_info(i), ax=axes[1])
            axes[0].title.set_text('Predicted')
            axes[1].title.set_text('Ground Truth')
            fig.suptitle("Dependence Graph Prediction")
            plt.show(block=False)
            plot_adj_mats(pred_adj, adj_mat, titles=['Predicted', 'Ground Truth'])
            plt.show()

        def sce_vis(i):
            visualize_scene(i)

        adj_vis(i, pred_adj, data.adj_mat[0])
        # sce_vis(i)
        # plt.show()


def main():
    feat_net = load_model(ObjectNet, 'cn_test_best_model.pt')
    feat_net.eval()
    train_loader, val_loader, test_loader = get_depdataloaders(feat_net)

    dep_net = load_model(DNet, 'dnT_best_model_GAT16.pt',
                         model_args=[511, 256, 128], model_kwargs={'heads': 16, 'concat':False})
    # dep_net = load_model(DNet, 'dnT_best_model_fixnosam_gat8noconcat.pt', model_args=[511, 256, 128])
    dep_net.eval()

    vis_test(dep_net, test_loader)


if __name__ == '__main__':
    main()
