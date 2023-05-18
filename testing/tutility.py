import numpy as np
import matplotlib.pyplot as plt


def print_dep(graph, node_ids, name):
    """
    graph is the dependence graph,
    node_ids is a list of object ids,
    name is a function which returns name from object id
    """
    for i, r in enumerate(graph):
        for j, c in enumerate(r):
            if c:
                nidi = int(node_ids[i])
                nidj = int(node_ids[j])
                print(f'{name(nidi)}[{nidi}] depends on {name(nidj)}[{nidj}]')


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
