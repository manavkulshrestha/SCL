import numpy as np
import pickle

from rp import dep_dict
from utility import std, mean
from toposort import toposort

last_0_10 = 'results_100_0-10_1685636907'
last_10_15 = 'results_100_10-15_1684864331.5446715'
# expert = 'results_expertrandom_100_0-10_1684878390.7436266'

was_legitmaybe = 'results_100_0-10_1684869182.5276263'

base_0_10 = 'results_iterative_732_0-10_1684976635'
base_10_15 = 'results_iterative_1000_10-15_1685149585'
base_15_20 = 'results_iterative_1000_15-20_1685241052'
base_20_25 = 'results_iterative_1000_20-25_1685279855'

rbase_0_10 = 'results_random_732_0-10_1684979952'
# rbase_10_15 = 'results_iterative_1000_10-15_1685149585'
# rbase_15_20 = 'results_iterative_1000_15-20_1685241052'
# rbase_20_25 = 'results_iterative_1000_20-25_1685279855'


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


def move_nums(results):
    move_times = [r['total_move_time'] for r in results]
    move_tries = [r['total_move_try_num'] for r in results]

    print(f'Expert planning time: {mean(move_times)}+/-{std(move_times)}')
    print(f'Expert planning tries: {mean(move_tries)}+/-{std(move_tries)}')
    nodes(results)


def get_valids(results):
    good_is = [r['data_idx'] for r in results]
    print(good_is)
    print(len(good_is))


def get_success_rate(results, cutoff_factor, title=''):
    obj_nums = np.array([r['num_nodes'] for r in results])
    steps_taken = np.array([r['total_move_try_num'] for r in results])

    successes = steps_taken < cutoff_factor*obj_nums

    pos_means = np.array([r['pos_mean'] for r in results])[successes].mean()
    pos_stds = np.array([r['pos_std'] for r in results])[successes].mean()
    orn_means = np.array([r['orn_mean'] for r in results])[successes].mean()
    orn_stds = np.array([r['orn_std'] for r in results])[successes].mean()

    completions = np.array([(np.cumsum(r['move_try_nums']) < cutoff_factor*r['num_nodes']).mean() for r in results])

    steps_factor = steps_taken[successes]/obj_nums[successes]

    print(f'{title} SUCCESS THRESHOLD ==> steps < {cutoff_factor}*[number of objects]')
    print('success rate:', successes.mean() * 100)
    print(f'completion rate: ', completions.mean() * 100)
    print(f'steps (multiplier): {steps_factor.mean()}+/-{np.std(steps_factor)}')
    print(f'pos error: {pos_means}+/-{pos_stds}')
    print(f'orn error: {orn_means}+/-{orn_stds}')
    print(f'average total steps taken: {steps_taken[successes].mean()}+/-{np.std(steps_taken[successes])}')


def get_num_obj_suc_exp(results):
    # with open(f'results/{last_0_10}', 'rb') as f:
    #     results = pickle.load(f)
    obj_nums = np.array([r['num_nodes'] for r in results])
    successes = np.array([r['success'] for r in results])

    met = obj_nums[successes]
    print(f'{met.mean()} +/- {np.std(met)}')


def ideal_para_split(results):
    successes = [r for r in results if r['success']]
    layerss = [list(toposort(dep_dict(r['pred_graph']))) for r in successes]

    def layers_man(layers, mans=1):
        return sum(len(layer)//mans+1 for layer in layers)

    steps_one = np.array([layers_man(layers) for layers in layerss])
    steps_two = np.array([layers_man(layers, mans=2) for layers in layerss])

    speed_ups = steps_one/steps_two
    print(speed_ups)
    print(f'Speed up factor: {speed_ups.mean()} +/ {np.std(speed_ups)}')


def main():
    # with open(f'results/classical/{base_0_10}', 'rb') as f:
    #     iresults = pickle.load(f)
    # with open(f'results/classical/{rbase_0_10}', 'rb') as f:
    #     rresults = pickle.load(f)
    with open(f'results/{last_0_10}', 'rb') as f:
        results = pickle.load(f)

    ideal_para_split(results)


if __name__ == '__main__':
    main()
