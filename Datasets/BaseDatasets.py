import os
import pickle

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from utility import compare_rows
import torch.nn.functional as F

from toposort import toposort


class MLPDataset(Dataset):
    def __init__(self, data_path, chunk=(0, 500), max_nobjs=10, flat=False):
        self.x = []
        self.y = []

        self.num_objects = 0
        with open(data_path, 'rb') as f:
            base_data = pickle.load(f)

        for i, data in zip(range(*chunk), base_data):
            flat_plan = np.hstack([list(layer) for layer in data['plan']])

            init_x = self._pad_rows(data['init_x'], max_nobjs)
            goal_x = self._pad_rows(data['goal_x'], max_nobjs)
            curr_x = init_x.clone()

            for next_obj_idx in flat_plan:
                step_x = torch.stack([curr_x.clone(), goal_x])
                self.x.append(step_x.view(-1) if flat else step_x)
                self.y.append(torch.tensor(next_obj_idx).cuda())

                curr_x[next_obj_idx] = goal_x[next_obj_idx]

    @staticmethod
    def _pad_rows(tensor, max_nobjs):
        n_objs = tensor.size(0)
        padded_tensor = F.pad(tensor, (0, 0, 0, max_nobjs - n_objs))
        return padded_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
