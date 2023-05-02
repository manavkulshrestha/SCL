from torch_geometric.data import Data, InMemoryDataset
from scipy.sparse import coo_matrix
import torch

import numpy as np

from os import path as osp

from tqdm import tqdm


class DependenceDataset(InMemoryDataset):
    def __init__(self, root, *, chunk, transform=None, pre_transform=None, pre_filter=None,
                 train=True):
        self.chunk = chunk

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'ddata_{self.chunk[0]}-{self.chunk[1]}.pt']

    def process_files(self):
        data_list = []

        for i in tqdm(range(*self.chunk), desc=f'Processing'):
            npzfile = np.load(osp.join(self.root, f'{i // 1000}_{i % 1000}.npz'))

            node_ids, nodes_feat, depg = [npzfile[x] for x in ['node_ids', 'nodes_feat', 'depg']]
            x = torch.tensor(nodes_feat, dtype=torch.float)

            # depg[i][j] == 1 --> i depends on j. edge labels are all ones
            adj = coo_matrix(depg)
            edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)

            # put into data. x is node features, edge_index is ground truth links' (row, col)
            data = Data(x=x, edge_index=edge_index, num_nodes=len(node_ids), depg=depg)
            data_list.append(data)

        return data_list

    def process(self):
        data_list = self.process_files()
        # print('datalist done')
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        # print('done with prefilter')

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # print('done with pretransform')
        data, slices = self.collate(data_list)
        # print('done collating')
        torch.save((data, slices), self.processed_paths[0])
        print('done saving')


class ClassificationDataset(InMemoryDataset):
    def __init__(self, root, *, chunk, transform=None, pre_transform=None, pre_filter=None,
                 train=True):
        self.chunk = chunk

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'cdata_{self.chunk[0]}-{self.chunk[1]}.pt']

    def process_files(self):
        data_list = []

        for i in tqdm(range(*self.chunk), desc=f'Processing'):
            npzfile = np.load(osp.join(self.root, f'{i // 1000}_{i % 1000}.npz'))
            pcds, oids, tids = [npzfile[x] for x in ['pc', 'oid', 'tid']]

            oid_tid_map = dict(zip(oids, tids))
            for oid in enumerate(np.unique(oids)):
                obj_pcd = pcds[oids == oid]
                tid = oid_tid_map[oid]

                data = Data(x=obj_pcd, y=tid)
                data_list.append(data)

        return data_list

    def process(self):
        data_list = self.process_files()
        # print('datalist done')
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        # print('done with prefilter')

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # print('done with pretransform')
        data, slices = self.collate(data_list)
        # print('done collating')
        torch.save((data, slices), self.processed_paths[0])
        print('done saving')