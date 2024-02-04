
import os
import json
from tqdm import tqdm
import numpy as np
import random
import logging
import argparse
import torch
from multiprocessing import cpu_count
from itertools import permutations
from multiprocessing.pool import ThreadPool
from mmlu_categories import categories, subject2cat

logger = logging.getLogger(__name__)

BAD_OPTIONS = [
    'A and B',
    'A and C',
    'A and D',
    'A nor B',
    'A or B',
    'B and C',
    'B and D',
    'C and D',
]
REFER_OPTIONS = [
    'all of the above',
    'all of above',
    'none of the above',
    'none of above',
]


def save_results(save_file_path, results: list, metrics: dict = None):
    while True:
        try:
            with open(save_file_path, 'w') as f:
                for result in results + ([metrics] if metrics is not None else []):
                    f.write(json.dumps(result) + '\n')
            break
        except OSError as e:
            logger.info(f"OSError: {e}. Retrying.")
            continue


def load_results(
    load_path, by_category=None,
    only_probs_and_ideals=False,
    skip_bad_options=True,
    skip_cot_outliers=True,
):
    if by_category is not None:
        assert by_category in categories

    subjects = []
    records = {}
    record_files = sorted(os.listdir(load_path))
    for record_file in record_files:
        if not record_file.endswith('.jsonl'):
            continue
        subject = record_file[:-6]
        if by_category is not None and subject2cat[subject] != by_category:
            continue
        record_path = f'{load_path}/{record_file}'
        record_lines = [json.loads(line) for line in open(record_path)]
        record_lines = [line for line in record_lines if line['type'] == 'result']
        record_lines = sorted(record_lines, key=lambda x: x['data']['idx'])
        subjects.append(subject)
        records[subject] = record_lines

    subjects = sorted(subjects)
    predictions = []
    labels = []
    cnt = [0, 0, 0]
    for subject in subjects:
        record_lines = records[subject]
        for line in record_lines:
            cnt[0] += 1
            if skip_bad_options and (
                any(e in option for e in BAD_OPTIONS for option in line['data']['options']) or
                any(e in option.lower() for e in REFER_OPTIONS for option in line['data']['options'])
            ):
                cnt[1] += 1
                continue
            if skip_cot_outliers and (
                'cot' in load_path and not any(line['data']['sampled'].startswith(e) for e in 'ABCD')
            ):
                cnt[2] += 1
                continue

            sample_id = (subject, int(line['data']['idx']))
            if only_probs_and_ideals:
                prob_name = 'observed' if 'ichat' in load_path else 'probs'
                predictions.append(line['data'][prob_name])
            else:
                if 'ichat' in load_path and 'noid' in load_path:
                    if line['data']['correct']:
                        predictions.append(line['data']['ideal'])
                    else:
                        try:
                            predictions.append('ABCDE'[
                                [_norm(e) for e in line['data']['options']].index(
                                    _norm(line['data']['sampled']))
                            ])
                        except ValueError:
                            predictions.append('Z')
                else:
                    predictions.append(line['data']['sampled'][0])
            labels.append(line['data']['ideal'])

    if only_probs_and_ideals:
        return list(zip(predictions, labels))
    else:
        return predictions, labels


def cycle_options(answers):
    n = len(answers)
    for i in range(n):
        cycled_answers = answers[i:] + answers[:i]
        yield cycled_answers


def permute_options(answers):
    permuted_indices = list(sorted(permutations(range(len(answers)))))
    for permuted_index in permuted_indices:
        permuted_answers = [answers[i] for i in permuted_index]
        yield permuted_answers


def shuffle_options_with_ids(option_ids, options):
    seed = '\n'.join(list(map(str, options))).encode("utf-8")
    rng = random.Random(seed)
    option_ids_and_options = list(zip(option_ids, options))
    rng.shuffle(option_ids_and_options)
    option_ids = [e[0] for e in option_ids_and_options]
    options = [e[1] for e in option_ids_and_options]
    return option_ids, options


def move_answer(x, moved_option):
    if x["Answer"] == moved_option:
        return x
    else:
        x[x["Answer"]], x[moved_option] = x[moved_option], x[x["Answer"]]
        x["Answer"] = moved_option
        return x


def eval_all_samples(eval_fn, eval_samples, name=None, max_num_samples=None, existing_results=None, threads=10):
    work_items = _index_samples(eval_samples, max_num_samples)

    if existing_results is not None:
        broken_results = [e for e in existing_results if e['type'] == 'broken']
        existing_results = [e for e in existing_results if e['type'] == 'result']
        if len(broken_results) == 0:
            return existing_results
        existing_results_ids = set([e['data']['idx'] for e in existing_results])
        work_items = [e for e in work_items if e[0] not in existing_results_ids]
    else:
        existing_results = []

    def eval_sample(args):
        """
        Evaluate a single sample.
        """
        idx, _ = args
        seed = f"{idx}:20230101".encode("utf-8")
        rng = random.Random(seed)
        return eval_fn(args, rng)

    while True:
        try:
            with ThreadPool(threads) as pool:
                if threads > 1:
                    logger.info(f"Running in threaded mode with {threads} threads!")
                    iter = pool.imap_unordered(eval_sample, work_items)
                else:
                    logger.info(f"Running in sequential mode!")
                    iter = map(eval_sample, work_items)
                results = list(tqdm(iter, total=len(work_items), dynamic_ncols=True, desc=name))
            break

        except RuntimeError as e:
            if threads > 1:
                threads = threads // 2
                logger.info(f"RuntimeError: {e}. Retrying with {threads} threads.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    results = sorted(existing_results + results, key=lambda x: x['data']['idx'])
    return results


def _index_samples(samples, max_num_samples=None, remaining=False):
    indices = list(range(len(samples)))
    random.Random(123).shuffle(indices)
    if max_num_samples is not None:
        if remaining:
            indices = indices[max_num_samples:]
        else:
            indices = indices[:max_num_samples]
    logger.info(f"Evaluating {len(indices)} samples")
    work_items = [(idx, samples[idx]) for idx in sorted(indices)]
    return work_items


def get_accuracy(data):
    results = [int(e['data']["correct"]) for e in data if e['type'] == 'result']
    num_correct = sum(results)
    num_total = len(results)
    if num_total == 0:
        return float("nan")
    else:
        return num_correct / num_total


def get_bootstrap_accuracy_std(data, num_samples=1000):
    rng = random.Random(123)
    vals = [e['data']["correct"] for e in data if e['type'] == 'result']
    return np.std([np.mean(rng.sample(vals, len(vals) // 2)) for _ in range(num_samples)])


def _norm(x):
    return ' '.join(x.strip().split())


def chunklist(lst, n):
    avg = len(lst) // n
    remainder = len(lst) % n
    chunks = []
    start = 0
    for i in range(n):
        end = start + avg + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def patch_open():
    import builtins
    import io

    prev_open = open

    def new_open(*args, **kwargs):
        buffer_size = kwargs.pop("buffering", io.DEFAULT_BUFFER_SIZE)
        kwargs["buffering"] = min(io.DEFAULT_BUFFER_SIZE, buffer_size)
        return prev_open(*args, **kwargs)

    builtins.open = new_open


def _purple(str: str) -> str:
    return f"\033[1;35m{str}\033[0m"
def _orange(str: str) -> str:
    return f"\033[1;31m{str}\033[0m"
def _blue(str: str) -> str:
    return f"\033[1;34m{str}\033[0m"
