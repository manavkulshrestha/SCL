import numpy as np
import pickle


last_0_10 = 'results_100_0-10_1684862573.1763673'
last_10_15 = 'results_100_10-15_1684864331.5446715'


def recovery(results):
    pred_wrong = list(filter(lambda r: not r['graphs_equal'], results))
    pred_wrong_succ = list(filter(lambda r: r['success'], pred_wrong))

    print(f'{len(pred_wrong_succ)}/{len(pred_wrong)} = {len(pred_wrong_succ) / len(pred_wrong)}')


def main():
    with open('results/results_100_0-10_1684869182.5276263', 'rb') as f:
        results = pickle.load(f)

    fails = filter(lambda r: not r['success'] and r['data_num'] < 100, results)
    print([f['data_idx'] for f in fails])



if __name__ == '__main__':
    main()