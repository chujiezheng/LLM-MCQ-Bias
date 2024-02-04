
import os
import json
import math
import argparse
import random
import numpy as np
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
import multiprocessing as mp
from sklearn.metrics import classification_report
from utils import load_results
from debias_utils import simple as debias_fn

TASKS = {
    'mmlu': 'MMLU',
    'arc': 'ARC',
    'csqa': 'CSQA',
}
NUM_SHOTS = [
    0,
    5,
]
MODELS = {
    "llama-7b": ('llama', 7),
    "llama-13b": ('llama', 13),
    "llama-30b": ('llama', 30),
    "llama-65b": ('llama', 65),
    "Llama-2-7b-hf": ('llama-2', 7),
    "Llama-2-13b-hf": ('llama-2', 13),
    "Llama-2-70b-hf": ('llama-2', 70),
    "Llama-2-7b-chat-hf": ('llama-2-chat', 7),
    "Llama-2-13b-chat-hf": ('llama-2-chat', 13),
    "Llama-2-70b-chat-hf": ('llama-2-chat', 70),
    "vicuna-7b-v1.3": ('vicuna-v1.3', 7),
    "vicuna-13b-v1.3": ('vicuna-v1.3', 13),
    "vicuna-33b-v1.3": ('vicuna-v1.3', 33),
    "vicuna-7b-v1.5": ('vicuna-v1.5', 7),
    "vicuna-13b-v1.5": ('vicuna-v1.5', 13),
    "falcon-7b": ('falcon', 7),
    "falcon-40b": ('falcon', 40),
    "falcon-7b-instruct": ('falcon-inst', 7),
    "falcon-40b-instruct": ('falcon-inst', 40),
    "ichat": ('gpt-3.5-turbo', 130),
}

SAVE_PATH = 'debias_pride'
os.makedirs(SAVE_PATH, exist_ok=True)
RATIO_PREFIX_SAMPLES = 0.05

TRANSFER = False


task_pairs = [(task, task) for task in TASKS]
if TRANSFER:
    subtasks = [
        'STEM',
        'Social Science',
        'Humanities',
        'Others',
        'arc',
    ]
    task_pairs = [(x, y) for x in subtasks for y in subtasks]

if TRANSFER:
    record_file = f'{SAVE_PATH}/debias_{RATIO_PREFIX_SAMPLES}_transfer.json'
else:
    record_file = f'{SAVE_PATH}/debias_{RATIO_PREFIX_SAMPLES}.json'


def single_process(args):
    num_shots, (source_task, target_task), (model, (model_family, model_size)) = args
    model_name = model_family
    if model != 'ichat':
        model_name += f'-{model_size}B'

    if not TRANSFER:
        assert source_task == target_task

    by_category = None if source_task in TASKS else source_task
    task = source_task
    if TRANSFER and source_task != 'arc':
        task = 'mmlu'
    source_path = f'results_{task}/{num_shots}s_{model}/{task}_perm'
    if not os.path.exists(source_path):
        source_path = f'results_{task}/{num_shots}s_{model}/{task}_cyclic'
        if not os.path.exists(source_path):
            return None
    all_source_probs_and_ideals = load_results(source_path, by_category=by_category, only_probs_and_ideals=True)
    source_rng = random.Random(source_path.encode('utf-8'))

    if TRANSFER:
        by_category = None if target_task in TASKS else target_task
        task = target_task
        if TRANSFER and target_task != 'arc':
            task = 'mmlu'
        target_path = f'results_{task}/{num_shots}s_{model}/{task}_perm'
        if not os.path.exists(target_path):
            target_path = f'results_{task}/{num_shots}s_{model}/{task}_cyclic'
            if not os.path.exists(target_path):
                return None
        all_target_probs_and_ideals = load_results(target_path, by_category=by_category, only_probs_and_ideals=True)

    n_iters = 5
    if RATIO_PREFIX_SAMPLES == 1.:
        n_iters = 1

    num_prefix_samples = int(len(all_source_probs_and_ideals) * RATIO_PREFIX_SAMPLES)

    costs = []
    scores = []
    recall_stds = []
    for iter_idx in range(n_iters):
        predictions = []
        labels = []
        cost = []

        source_rng.shuffle(all_source_probs_and_ideals)
        prefix_samples = all_source_probs_and_ideals[:num_prefix_samples]
        if not TRANSFER:
            postfix_samples = all_source_probs_and_ideals[num_prefix_samples:]
        else:
            postfix_samples = all_target_probs_and_ideals[:]

        all_priors = []
        all_observed = []
        for idx, prefix_sample in enumerate(prefix_samples):
            observed, ideal = prefix_sample
            observed = np.array(observed)
            observed, debiased, prior = debias_fn(observed)
            all_priors.append(prior)
            all_observed.append(observed)
            predictions.append(np.argmax(debiased))
            cost.append(len(observed))
            labels.append('ABCDE'.index(ideal))

        prior = np.mean(all_priors, axis=0)

        for postfix_sample in postfix_samples:
            observed, ideal = postfix_sample
            observed = np.array(observed[0])
            debiased = np.log(observed + 1e-10) - np.log(prior + 1e-10)
            predictions.append(np.argmax(debiased))
            cost.append(1)
            labels.append('ABCDE'.index(ideal))

        final_score = np.mean(np.array(predictions) == np.array(labels)) * 100
        scores.append(final_score)
        costs.append(np.mean(cost))

        report = classification_report(labels, predictions, output_dict=True)
        recalls = [report[str(e)]['recall'] * 100 for e in range(prior.shape[-1])]
        recall_stds.append(np.std(recalls))

    res = {
        'num_shots': num_shots,
        'source_task': source_task,
        'target_task': target_task,
        'model': model,
        'rstd': float(np.mean(recall_stds)),
        'rstd_max': float(np.max(recall_stds)),
        'rstd_min': float(np.min(recall_stds)),
        'rstd_std': float(np.std(recall_stds)),
        'acc': float(np.mean(scores)),
        'acc_max': float(np.max(scores)),
        'acc_min': float(np.min(scores)),
        'acc_std': float(np.std(scores)),
        'cost': float(np.mean(costs)),
    }
    return res


def main():
    args_list = []
    for task_pair in task_pairs:
        for num_shots in NUM_SHOTS:
            for model in MODELS:
                args_list.append((num_shots, task_pair, (model, MODELS[model])))

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(single_process, args_list), total=len(args_list), dynamic_ncols=True))

    all_results = []
    for result in results:
        if result is None:
            continue
        all_results.append(result)
    with open(record_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
