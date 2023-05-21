from torch_geometric.data import Data, InMemoryDataset
from scipy.sparse import coo_matrix
import torch

import numpy as np

from os import path as osp

from tqdm import tqdm

from nn.PositionalEncoder import PositionalEncoding
from utility import all_edges


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


class GraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['all_e_idx', 'gt_e_idx']:
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['all_e_idx', 'gt_e_idx']:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class DependenceDataset(InMemoryDataset):
    def __init__(self, pcd_root, dep_root, *, feat_net, chunk,
                 transform=None, pre_transform=None, pre_filter=None, sample_count=None):
        self.chunk = chunk
        self.sample_count = sample_count
        self.pcd_root = pcd_root
        self.feat_net = feat_net
        self.feat_net.eval()
        self.pos_enc = PositionalEncoding(min_deg=0, max_deg=5, scale=1, offset=0)

        super().__init__(dep_root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'ddata2_nsl_new_{self.chunk[0]}-{self.chunk[1]}.pt']

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

            # depg[i][j] == 1 --> i depends on j. edge labels are all ones. edges are of form (row, col)
            adj = coo_matrix(dep_g)
            gt_e_idx = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
            all_e_idx = all_edges(len(node_ids))
            all_e_y = torch.tensor(dep_g[tuple(all_e_idx)]).view(-1)

            # x is nodes' features, gt_e_idx is ground truth links', all_e_y are labels for all_e_idx
            data = GraphData(x=nodes_feats.cpu(), gt_e_idx=gt_e_idx, all_e_idx=all_e_idx, all_e_y=all_e_y,
                             num_nodes=len(node_ids), adj_mat=dep_g)
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


class AllDataset(InMemoryDataset):
    def __init__(self, root, *, feat_net, chunk,
                 transform=None, pre_transform=None, pre_filter=None, sample_count=None):
        self.chunk = chunk
        self.sample_count = sample_count
        self.root = root
        self.feat_net = feat_net
        self.feat_net.eval()
        self.pos_enc = PositionalEncoding(min_deg=0, max_deg=5, scale=1, offset=0)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'adata_{self.chunk[0]}-{self.chunk[1]}.pt']

    def process_files(self):
        data_list = []

        for i in tqdm(range(*self.chunk), desc=f'Processing'):
            file_name = f'{i // 1000}_{i % 1000}.npz'
            all_file = np.load(osp.join(self.root, file_name))
            to_extract = ['pc', 'oid', 'tid', 'depg', 'pos', 'orn', 'node_ids']
            pcds, o_ids, t_ids, dep_g, pos, orn, node_ids = [all_file[x] for x in to_extract]

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

            # depg[i][j] == 1 --> i depends on j. edge labels are all ones. edges are of form (row, col)
            adj = coo_matrix(dep_g)
            gt_e_idx = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
            all_e_idx = all_edges(len(node_ids))
            all_e_y = torch.tensor(dep_g[tuple(all_e_idx)]).view(-1)

            # x is nodes' features, gt_e_idx is ground truth links', all_e_y are labels for all_e_idx
            data = GraphData(x=nodes_feats.cpu(), gt_e_idx=gt_e_idx, all_e_idx=all_e_idx, all_e_y=all_e_y,
                             node_ids=node_ids, num_nodes=len(node_ids), adj_mat=dep_g,
                             g_poss=pos, g_orns=orn, o_ids=o_ids, t_ids=t_ids)
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
