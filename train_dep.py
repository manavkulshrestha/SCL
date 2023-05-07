import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter

import os.path as osp

import numpy as np

import tqdm

from utility import get_depdataloaders, get_rowcol, load_depgs, coo_withzeros, coo_matrix, save_model,\
                    device, DDPATH, RP_ROOT
from Network import DependenceNet

writer = SummaryWriter(log_dir='logs')

def loss_fn(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


def train_epoch(model, epoch, loader, optimizer, progress=False):
    model.train()
    train_loss = 0.0

    progress = tqdm if progress else lambda x, **kwargs: x

    total_examples = 0

    for batch in progress(loader, desc=f'[Epoch {epoch:03d}] training'):
        batch = batch.to(device)
        optimizer.zero_grad()

        # negative sampling for class imbalance
        z = model.encode(batch.x, batch.edge_index)
        nedge_index = negative_sampling(batch.edge_index, batch.num_nodes)  # sample non-edges
        edge_label_index = torch.cat([batch.edge_index, nedge_index], dim=-1).to(device)
        edge_label = torch.cat(
            [torch.ones(batch.edge_index.size(1)), torch.zeros(nedge_index.size(1))], dim=0
        ).to(device)
        out = model.decode(z, edge_label_index).view(-1)

        # get loss and update model
        loss = loss_fn(out, edge_label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
        total_examples += batch.num_graphs

    train_loss /= total_examples
    return train_loss


@torch.no_grad()
def test_epoch(model, epoch, loader, thresh=0.9, progress=False):
    model.eval()
    scores = []
    threshold = torch.tensor(thresh)

    progress = tqdm if progress else lambda x, **kwargs: x

    for i, batch in enumerate(progress(loader, desc=f'[Epoch {epoch:03d}] testing')):
        batch = batch.to(device)
        edge_label_index, edge_label = coo_withzeros(i, batch.num_nodes, batch.edge_index.cpu().numpy(), batch.depg)
        edge_label_index = torch.tensor(edge_label_index).to(device)

        # encode all edges #TODO check
        z = model.encode(batch.x, edge_label_index)
        out = model.decode(z, edge_label_index).view(-1).sigmoid()
        pred = (out > threshold).float()

        score = f1_score(edge_label, pred.cpu().numpy())
        scores.append(score)

    return np.mean(scores)


def main():
    train_loader, val_loader, test_loader = get_depdataloaders()

    model = DependenceNet(21, 512, 1024, 512, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    best_val_f1 = 0
    for epoch in range(1, 201):
        # get train loss and f1
        train_loss = train_epoch(model, epoch, train_loader, optimizer)
        val_f1 = test_epoch(model, epoch, val_loader)

        print(f'[Epoch {epoch:03d}] Train Loss: {train_loss:.4f}, Val F1: {val_f1}')

        # save best model based on validation f1
        if best_val_f1 < val_f1:
            best_val_f1 = val_f1
            save_model(model, 'dnT_best_model.pt')

        # save a model every 20 epochs
        if epoch % 20 == 0:
            save_model(model, f'dnT_model-{epoch}.pt')


if __name__ == '__main__':
    main()