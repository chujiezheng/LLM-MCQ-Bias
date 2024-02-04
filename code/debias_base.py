
import os
import json
import math
import argparse
import numpy as np
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from mmlu_categories import categories, subject2cat
from utils import get_accuracy, get_bootstrap_accuracy_std, save_results


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--debias_fn", type=str, required=True)
parser.add_argument("--load_paths", type=str, nargs='+', default=[])
args = parser.parse_args()

if len(args.load_paths) == 0:
    exit()

if args.task not in [
    'mmlu', 'arc', 'csqa'
]:
    raise ValueError(f'Unknown task: {args.task}')

if args.debias_fn == 'simple':
    from debias_utils import simple as debias_fn
elif args.debias_fn == 'full':
    from debias_utils import full as debias_fn
else:
    raise ValueError(f'Unknown debias function: {args.debias_fn}')


def single_process(load_path):
    if load_path[-1] == '/':
        load_path = load_path[:-1]
    if 'perm' not in load_path and 'cyclic' not in load_path:
        logger.info(f'{load_path} does not contain perm or cyclic')
        return 1

    if 'perm' in load_path:
        source = 'perm'
        assert args.debias_fn in [
            'simple', 'full',
        ]
    else:
        source = 'cyclic'
        assert args.debias_fn in [
            'simple',
        ]

    save_path = load_path.replace(source, args.debias_fn)
    os.makedirs(save_path, exist_ok=True)

    by_category_list = [None]
    #if 'mmlu' in args.task:
    #    by_category_list += list(categories.keys())

    for by_category in by_category_list:
        try:
            priors = debias_results(
                debias_fn, load_path,
                by_category=by_category,
                save_path=save_path if by_category is None else None,
            )
        except ValueError as e:
            logger.info(f'Failed to calibrate {load_path}')
            logger.info(f'Error: {e}')
            raise e

    return 0


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(single_process, args.load_paths), total=len(args.load_paths), dynamic_ncols=True))


def debias_results(debias_fn, load_path, save_path=None, by_category=None):
    if by_category is not None:
        assert by_category in categories

    prob_name = 'observed' if 'ichat' in load_path else 'probs'
    record_files = sorted(os.listdir(load_path))

    all_priors = []
    for record_file in record_files:
        if not record_file.endswith('.jsonl'):
            continue
        subject = record_file[:-6]
        if by_category is not None and subject2cat[subject] != by_category:
            continue
        record_path = f'{load_path}/{record_file}'
        data = [json.loads(line) for line in open(record_path)]
        data = [line for line in data if line['type'] == 'result']

        for d in data:
            observed, debiased, prior = debias_fn(d['data'][prob_name])
            d['data']['sampled'] = 'ABCDE'[np.argmax(debiased)]
            d['data']['correct'] = (d['data']['sampled'] == d['data']['ideal'])
            all_priors.append(prior.tolist())
            if 'prompt' in d['data']:
                d['data'].pop('prompt')
            d['data'][prob_name] = observed.tolist()

        if save_path is not None:
            metrics = {'type': 'metric', 'data': {}}
            metrics['data']['accuracy'] = get_accuracy(data)
            metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(data)
            save_results(f'{save_path}/{record_file}', data, metrics)

    return all_priors


if __name__ == '__main__':
    main()
