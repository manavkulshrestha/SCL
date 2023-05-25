from Datasets.dutility import get_alldataloaders, get_scenesdataloader
from nn.Network import ObjectNet
from utility import load_model


def condition(data, l_bound, u_bound):
    return l_bound < len(data.adj_mat[0]) <= u_bound


def main():
    # load models
    feat_net = load_model(ObjectNet, 'cn_test_best_model.pt')
    feat_net.eval()

    # load data and do experiments
    count = 0
    _, _, test_loader = get_scenesdataloader(feat_net)
    print('done loading everything')
    for i, data in enumerate(test_loader):
        count += 1 if condition(data, 0, 10) else 0

    print(f'{count}/{len(test_loader)} data points satisfy condition')


if __name__ == '__main__':
    main()
