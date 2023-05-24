import numpy as np
import pickle

from utility import std, mean

last_0_10 = 'results_100_0-10_1684862573.1763673'
last_10_15 = 'results_100_10-15_1684864331.5446715'
expert = 'results_expertrandom_100_0-10_1684878390.7436266'

was_legitmaybe = 'results_100_0-10_1684869182.5276263'


def recovery(results):
    pred_wrong = list(filter(lambda r: not r['graphs_equal'], results))
    pred_wrong_succ = list(filter(lambda r: r['success'], pred_wrong))

    print(f'{len(pred_wrong_succ)}/{len(pred_wrong)} = {len(pred_wrong_succ) / len(pred_wrong)}')


def failure(results):
    fails = filter(lambda r: not r['success'] and r['data_num'] < 100, results)
    print([f['data_idx'] for f in fails])


def nodes(results):
    num_nodes = [r['num_nodes'] for r in results]
    print(f'{mean(num_nodes)}+/-{std(num_nodes)}, min: {min(num_nodes)}, max: {max(num_nodes)}')


def main():
    with open(f'results/{expert}', 'rb') as f:
        results = pickle.load(f)

    move_times = [r['total_move_time'] for r in results]
    move_tries = [r['total_move_try_num'] for r in results]

    print(f'Expert planning time: {mean(move_times)}+/-{std(move_times)}')
    print(f'Expert planning tries: {mean(move_tries)}+/-{std(move_tries)}')
    nodes(results)


if __name__ == '__main__':
    main()