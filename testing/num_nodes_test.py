import numpy as np
from torch_geometric.data import DataLoader

from Datasets.Dataset import DependenceDataset
from nn.Network import ObjectNet
from Datasets.dutility import PDPATH, DDPATH

import torch
import torch_geometric.transforms as T


def main():
    feat_net = ObjectNet().cuda()
    feat_net.load_state_dict(torch.load('/home/mk/rp/models/cn_test_best_model.pt'))
    feat_net.eval()

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice('cuda'),
    ])

    sc = 512
    train_dataset = DependenceDataset(PDPATH, DDPATH, feat_net=feat_net, chunk=(0, 8000), transform=transform, sample_count=sc)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    nums_nodes = np.array([b.num_nodes for b in train_loader])
    print(f'mean: {nums_nodes.mean()}, median: {np.median(nums_nodes)}, std: {np.std(nums_nodes)}, min: {nums_nodes.min()}, max: {nums_nodes.max()}')


if __name__ == '__main__':
    main()