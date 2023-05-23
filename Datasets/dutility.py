from Datasets.Dataset import DependenceDataset, ObjectDataset, AllDataset
from utility import RP_ROOT
import os.path as osp
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

DDPATH = osp.join(RP_ROOT, 'data/dep_data/')
PDPATH = osp.join(RP_ROOT, 'data/pcd_data/')
ADPATH = osp.join(RP_ROOT, 'data/all_data/')


def get_depdataloaders(feat_net, chunks=((0, 8000), (8000, 9000), (9000, 10000)),
                       batch_sizes=(64, 64, 1), shuffles=(True, True, False)):
    transform = T.Compose([
        # T.NormalizeFeatures(),
        T.ToDevice('cuda'),
    ])

    sc = 512
    train_dataset = DependenceDataset(PDPATH, DDPATH, feat_net=feat_net, chunk=chunks[0], transform=transform, sample_count=sc)
    val_dataset = DependenceDataset(PDPATH, DDPATH, feat_net=feat_net, chunk=chunks[1], transform=transform, sample_count=sc)
    test_dataset = DependenceDataset(PDPATH, DDPATH, feat_net=feat_net, chunk=chunks[2], transform=transform, sample_count=sc)

    train_loader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=shuffles[0])  # num workers causes error
    val_loader = DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=shuffles[1])  # num workers causes error
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=shuffles[2])  # num workers causes error

    return None, None, test_loader


def get_objdataloaders():
    # transform = T.Compose([
    #     T.NormalizeFeatures(),
    #     T.ToDevice(device),
    # ])

    # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    pre_transform, transform = T.NormalizeScale(), None

    sc = 512
    train_dataset = ObjectDataset(PDPATH, chunk=(0, 500), pre_transform=pre_transform, transform=transform, sample_count=sc)
    val_dataset = ObjectDataset(PDPATH, chunk=(500, 550), pre_transform=pre_transform, transform=transform, sample_count=sc)
    test_dataset = ObjectDataset(PDPATH, chunk=(550, 600), pre_transform=pre_transform, transform=transform, sample_count=sc)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # num workers causes error
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)  # num workers causes error
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # num workers causes error

    return train_loader, val_loader, test_loader


def get_alldataloaders(feat_net, chunks=((0, 8000), (8000, 9000), (9000, 10000)),
                       batch_sizes=(64, 64, 1), shuffles=(True, True, False)):
    transform = T.Compose([
        T.ToDevice('cuda'),
    ])

    sc = 512
    # train_dataset = AllDataset(ADPATH, feat_net=feat_net, chunk=chunks[0], transform=transform, sample_count=sc)
    # val_dataset = AllDataset(ADPATH, feat_net=feat_net, chunk=chunks[1], transform=transform, sample_count=sc)
    test_dataset = AllDataset(ADPATH, feat_net=feat_net, chunk=chunks[2], transform=transform, sample_count=sc)

    # train_loader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=shuffles[0])  # num workers causes error
    # val_loader = DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=shuffles[1])  # num workers causes error
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=shuffles[2])  # num workers causes error

    return None, None, test_loader


def get_scenesdataloader(feat_net, chunks=((0, 8000), (8000, 9000), (9000, 10000)),
                       batch_sizes=(64, 64, 1), shuffles=(True, True, False)):
    transform = T.Compose([
        T.ToDevice('cuda'),
    ])

    sc = 512
    dataset = AllDataset(ADPATH, feat_net=feat_net, chunk=(0, 10000), transform=transform, sample_count=sc)
    # val_dataset = AllDataset(ADPATH, feat_net=feat_net, chunk=chunks[1], transform=transform, sample_count=sc)
    # test_dataset = AllDataset(ADPATH, feat_net=feat_net, chunk=chunks[2], transform=transform, sample_count=sc)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)  # num workers causes error
    # val_loader = DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=shuffles[1])  # num workers causes error
    # test_loader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=shuffles[2])  # num workers causes error

    return None, None, loader
