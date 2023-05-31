import torch
from torch.nn import BatchNorm2d, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv, PointNetConv, global_max_pool, MLP, radius, fps, GATv2Conv, GraphSAGE
from torch_geometric.nn.dense.linear import Linear
from torch import nn
from utility import sliding, sample_exact, normalize, device
from torch_geometric.data import Data

import torch.nn.functional as F


class DependenceNet(torch.nn.Module):
    def __init__(self, *layer_sizes):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(*inout) for inout in sliding(layer_sizes, 2)])
        self.activation = F.leaky_relu

    def encode(self, x, edge_index):
        for convi in self.convs[:-1]:
            x = convi(x, edge_index)
            x = F.leaky_relu(x)

        return self.convs[-1](x, edge_index)

    @classmethod
    def decode(cls, z, edge_label_index):
        node1, node2 = edge_label_index
        return (z[node1] * z[node2]).sum(dim=-1)

    @classmethod
    def decode_all(cls, z):
        return (z @ z.t() > 0).nonzero().t()


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class ObjectNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 8], dropout=0.5, norm=None)

    def forward(self, data, get_emb=False):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        out, emb = self.mlp(x, return_emb=True)
        outs = out.log_softmax(dim=-1)
        if get_emb:
            return outs, emb
        else:
            return outs

    def prediction(self, data):
        with torch.no_grad():
            outs = self.forward(data)
            pred = outs.max(1)[1].item()+1

            return pred

    @classmethod
    def make_data(cls, x, sample=512):
        with torch.no_grad():
            pos = torch.tensor(sample_exact(normalize(x), sample), dtype=torch.float).cuda()
            batch = torch.zeros(pos.shape[0], dtype=torch.int64).cuda()
            data = Data(pos=pos, batch=batch)

            return data

    def predict(self, x, sample=512):
        with torch.no_grad():
            data = self.make_data(x, sample=sample)
            pred = self.prediction(data)

            return pred

    def embed(self, x, sample=512, get_pred=False):
        with torch.no_grad():
            data = self.make_data(x, sample=sample)
            outs, emb = self.forward(data, get_emb=True)

            pred_tid = outs.max(1)[1].item() + 1

            return (pred_tid, emb) if get_pred else emb

    def predict_fromfeatures(self, emb, max_ax=0):
        with torch.no_grad():
            out = self.mlp.lins[-1](emb)
            outs = out.log_softmax(dim=-1)

            return outs.max(max_ax)[1].item() + 1


class DNEncoder(torch.nn.Module):
        def __init__(self, in_c, h_c, out_c, heads=8, concat=False):
            super().__init__()
            self.layer1 = GATv2Conv(in_c, h_c, heads=heads, concat=concat)
            self.layer2 = GATv2Conv(h_c, out_c, heads=heads, concat=concat)
            # self.layer3 = GATv2Conv(h_c, out_c, heads=heads, concat=concat)
            self.activation = LeakyReLU()

        def forward(self, x, edge_index):
            x = self.layer1(x, edge_index)
            x = self.activation(x)
            x = self.layer2(x, edge_index)
            # x = self.activation(x)
            # x = self.layer3(x, edge_index)

            return x


class DNDecoder(torch.nn.Module):
    def __init__(self, h_c, **kwargs):
        super().__init__()
        self.linear1 = Linear(2*h_c, h_c)
        self.linear2 = Linear(h_c, 1)
        # self.linear3 = Linear(h_c//2, 1)
        self.activation = LeakyReLU()

    def forward(self, z, edge_label_index):
        row, col = edge_label_index
        zc = torch.cat([z[row], z[col]], dim=1)

        zc = self.linear1(zc)
        zc = self.activation(zc)
        zc = self.linear2(zc)
        # zc = self.activation(zc)
        # zc = self.linear3(zc)

        return zc.view(-1)


class DNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super().__init__()
        self.encoder = DNEncoder(in_channels, hidden_channels, out_channels, **kwargs)
        self.decoder = DNDecoder(out_channels, **kwargs)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z, edge_index)

        return out


class GNEncoder(torch.nn.Module):
    def __init__(self, in_c, h_c, out_c):
        super().__init__()
        self.layer1 = GraphSAGE(in_c, h_c, 1)
        self.layer2 = GraphSAGE(h_c, out_c, 1)
        # self.layer3 = GCNConv(h_c, h_c)
        # self.layer4 = GCNConv(h_c, h_c)
        # self.layer5 = GraphSAGE(h_c, out_c, 1)
        self.activation = LeakyReLU()

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.activation(x)
        x = self.layer2(x, edge_index)
        # x = self.activation(x)
        # x = self.layer3(x, edge_index)
        # x = self.activation(x)
        # x = self.layer4(x, edge_index)
        # x = self.activation(x)
        # x = self.layer5(x, edge_index)

        return x


class GNDecoder(torch.nn.Module):
    def __init__(self, h_c, **kwargs):
        super().__init__()
        self.linear1 = Linear(2*h_c, h_c)
        self.linear2 = Linear(h_c, 1)
        # # self.linear3 = Linear(h_c//2, 1)
        self.activation = LeakyReLU()
        pass

    def forward(self, z, edge_label_index):
        row, col = edge_label_index
        zc = torch.cat([z[row], z[col]], dim=1)

        zc = self.linear1(zc)
        zc = self.activation(zc)
        zc = self.linear2(zc)
        # zc = self.activation(zc)
        # zc = self.linear3(zc)

        return zc.view(-1)

        # return (z[row] * z[col]).sum(dim=-1)


class GNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super().__init__()
        self.encoder = GNEncoder(in_channels, hidden_channels, out_channels, **kwargs)
        self.decoder = GNDecoder(out_channels, **kwargs)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z, edge_index)

        return out