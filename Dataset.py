from torch_geometric.data import Data, InMemoryDataset
from scipy.sparse import coo_matrix
import torch

import numpy as np

from os import path as osp

from tqdm import tqdm
import open3d as o3d

from PositionalEncoder import PositionalEncoding

# class DependenceDataset(InMemoryDataset):
#     def __init__(self, root, *, chunk, transform=None, pre_transform=None, pre_filter=None,
#                  train=True):
#         self.chunk = chunk
#
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def processed_file_names(self):
#         return [f'ddata_{self.chunk[0]}-{self.chunk[1]}.pt']
#
#     def process_files(self):
#         data_list = []
#
#         for i in tqdm(range(*self.chunk), desc=f'Processing'):
#             npzfile = np.load(osp.join(self.root, f'{i // 1000}_{i % 1000}.npz'))
#
#             node_ids, nodes_feat, depg = [npzfile[x] for x in ['node_ids', 'nodes_feat', 'depg']]
#             x = torch.tensor(nodes_feat, dtype=torch.float)
#
#             # depg[i][j] == 1 --> i depends on j. edge labels are all ones
#             adj = coo_matrix(depg)
#             edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
#
#             # put into data. x is node features, edge_index is ground truth links' (row, col)
#             data = Data(x=x, edge_index=edge_index, num_nodes=len(node_ids), depg=depg)
#             data_list.append(data)
#
#         return data_list
#
#     def process(self):
#         data_list = self.process_files()
#         # print('datalist done')
#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]
#         # print('done with prefilter')
#
#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]
#         # print('done with pretransform')
#         data, slices = self.collate(data_list)
#         # print('done collating')
#         torch.save((data, slices), self.processed_paths[0])
#         print('done saving')


class ObjectDataset(InMemoryDataset):
    def __init__(self, root, *, chunk, transform=None, pre_transform=None, pre_filter=None, sample_count=None,
                 train=True):
        self.chunk = chunk
        self.sample_count = sample_count

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'cdata_{self.chunk[0]}-{self.chunk[1]}.pt']

    def process_files(self):
        data_list = []

        for i in tqdm(range(*self.chunk), desc=f'Processing'):
            # '0_0.npz' in os.listdir('/home/mk/rp/data/pcd_data')
            npzfile = np.load(osp.join(self.root, f'{i // 1000}_{i % 1000}.npz'))
            pcds, oids, tids = [npzfile[x] for x in ['pc', 'oid', 'tid']]

            oid_tid_map = dict(zip(oids, tids))
            for oid in np.unique(oids):
                obj_pcd = pcds[oids == oid]
                tid = oid_tid_map[oid]

                if self.sample_count is not None:
                    try:
                        sampled_idx = np.random.choice(len(obj_pcd), self.sample_count, replace=False)
                    except ValueError as e:
                        sampled_idx = np.random.choice(len(obj_pcd), self.sample_count, replace=True)
                        # print(len(obj_pcd))
                    obj_pcd = obj_pcd[sampled_idx]

                data = Data(pos=torch.tensor(obj_pcd).float(), y=torch.tensor(tid-1, dtype=torch.long))
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

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'


class DependenceDataset(InMemoryDataset):
    def __init__(self, pcd_root, dep_root, *, feat_net, chunk, transform=None, pre_transform=None, pre_filter=None, sample_count=None,
                 train=True):
        self.chunk = chunk
        self.sample_count = sample_count
        self.pcd_root = pcd_root
        self.feat_net = feat_net
        self.pos_enc = PositionalEncoding(min_deg=0, max_deg=5, scale=1, offset=0)

        super().__init__(dep_root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'ddata2_{self.chunk[0]}-{self.chunk[1]}.pt']

    def process_files(self):
        data_list = []

        for i in tqdm(range(*self.chunk), desc=f'Processing'):
            file_name = f'{i // 1000}_{i % 1000}.npz'
            pcd_file = np.load(osp.join(self.pcd_root, file_name))
            dep_file = np.load(osp.join(self.root, file_name))
            pcds, o_ids, t_ids = [pcd_file[x] for x in ['pc', 'oid', 'tid']]
            node_ids, dep_g = [dep_file[x] for x in ['node_ids', 'depg']]

            nodes_feats = []
            oid_tid_map = dict(zip(o_ids, t_ids))
            for oid in node_ids:
                obj_pcd = pcds[o_ids == oid]
                obj_cen = obj_pcd.mean(axis=0)
                tid = oid_tid_map[oid]

                if self.sample_count is not None:
                    idx = np.random.choice(len(obj_pcd), self.sample_count, replace=len(obj_pcd) < self.sample_count)
                    obj_pcd = obj_pcd[idx]

                # get total features from positional encoding of centroid and object level features from network
                cen_ten = torch.tensor(obj_cen, dtype=torch.float).cuda()
                pred_tid, obj_emb = self.feat_net.embed(obj_pcd, get_pred=True)

                obj_emb = torch.squeeze(obj_emb)
                pos_emb = self.pos_enc(cen_ten)
                x = torch.cat([pos_emb, obj_emb])
                nodes_feats.append(x)
            nodes_feats = torch.stack(nodes_feats)

            # depg[i][j] == 1 --> i depends on j. edge labels are all ones
            adj = coo_matrix(dep_g)
            edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long) #TODO check

            # put into data. x is nodes' features, edge_index is ground truth links' (row, col)
            data = Data(x=nodes_feats.cpu(), edge_index=edge_index, num_nodes=len(node_ids), depg=dep_g)
            data_list.append(data)

        return data_list

    def process(self):
        data_list = self.process_files()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        print('Done saving data set.')
