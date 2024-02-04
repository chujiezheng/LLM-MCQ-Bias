
import os
import sys
import random
import copy
import json
import argparse
import logging
from tqdm import tqdm
from typing import List
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    _norm,
    shuffle_options_with_ids,
    move_answer,
    cycle_options,
    permute_options,
)

logger = logging.getLogger(__name__)


def parse_arguments():
    logger.info(f'cuda is available {torch.cuda.is_available()}')
    logger.info(f'cuda device count {torch.cuda.device_count()}')
    logger.info(f'cuda device name {torch.cuda.get_device_name()}')

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--eval_names", type=str, nargs='+', default=[],
                        help='eval tasks and settings')
    args = parser.parse_args()

    args.model_name = args.pretrained_model_path.split('/')[-1]

    for eval_name in args.eval_names:
        eval_args = eval_name.split(',')
        task = eval_args[0]
        if task not in [
            'mmlu', 'arc', 'csqa',
        ]:
            raise ValueError(f"Unknown task: {task}")

        num_few_shot = int(eval_args[1])

        setting = eval_args[2] if len(eval_args) > 2 else None
        if setting is not None and not (
            setting in [
                'noid',
                'perm', 'cyclic',
                'shuffle_both',
            ] or (setting.startswith('move') and setting[-1] in ['a', 'b', 'c', 'd'])
        ):
            raise ValueError(f"Unknown setting: {setting}")

    return args


def prepare_eval(args, eval_name):
    # task and setting
    eval_args = eval_name.split(',')
    args.task = task = eval_args[0]
    args.num_few_shot = num_few_shot = int(eval_args[1])
    args.setting = setting = eval_args[2] if len(eval_args) > 2 and eval_args[2] else None
    if setting is not None and setting.startswith('move'):
        moved_answer = setting[-1].upper()

    # save_path
    save_path = f'results_{task}/{num_few_shot}s_{args.model_name}/{task}'
    if setting is not None:
        save_path += f'_{setting}'
    args.save_path = save_path
    os.makedirs(args.save_path, exist_ok=True)

    option_ids = list('ABCD')
    option_ids_header = list('ABCD')
    if task in ['csqa']:
        option_ids = list('ABCDE')
        option_ids_header = list('ABCDE')

    data_path = f'data_{task}'
    subjects = sorted([f.split("_test.csv")[0]
                       for f in os.listdir(f'{data_path}/test') if "_test.csv" in f])

    # sys_msg
    if 'mmlu' in task:
        sys_msg = 'The following are multiple choice questions about {}.'
    else: # task in ['arc', 'tqa']
        sys_msg = 'The following are multiple choice questions.'

    sys_msg += ' You should directly answer the question by choosing the correct option.'

    # create_user_prompt
    def create_user_prompt(question: str, options: List[str]):
        if setting in ['noid']:
            user_prompt = f"Question: {question.strip()}\nOptions:\n" + \
                "\n".join([f"{answer}".strip()
                           for option_id, answer in zip(option_ids, options)]) + \
                "\nAnswer:"
        elif setting in ['shuffle_both']:
            shuffled_option_ids, shuffled_options = shuffle_options_with_ids(option_ids, options)
            user_prompt = f"Question: {question.strip()}\nOptions:\n" + \
                "\n".join([f"{option_id}. {answer}".strip()
                           for option_id, answer in zip(shuffled_option_ids, shuffled_options)]) + \
                "\nAnswer:"
        else:
            user_prompt = f"Question: {question.strip()}\nOptions:\n" + \
                "\n".join([f"{option_id}. {answer}".strip()
                           for option_id, answer in zip(option_ids, options)]) + \
                "\nAnswer:"
        return user_prompt

    # prepare_few_shot_samples
    def prepare_few_shot_samples(subject):
        df = pd.read_csv(f'{data_path}/dev/{subject}_dev.csv', names=("Question", *option_ids_header, "Answer"), dtype=str)
        if setting in ['noid']:
            few_shot_samples = df.apply(lambda x:
                create_user_prompt(x["Question"], [x[e] for e in option_ids_header])
                + ' ' + x[x["Answer"]]
            , axis=1).to_list()
        else:
            few_shot_samples = df.apply(lambda x:
                create_user_prompt(x["Question"], [x[e] for e in option_ids_header])
                + ' ' + option_ids[option_ids_header.index(x["Answer"])]
            , axis=1).to_list()
        return few_shot_samples

    # prepare_eval_samples
    def prepare_eval_samples(subject):
        df = pd.read_csv(open(f'{data_path}/test/{subject}_test.csv'), names=("Question", *option_ids_header, "Answer"), dtype=str)

        if setting is not None and setting.startswith('move'):
            df = df.apply(lambda x: move_answer(x, moved_answer), axis=1)

        if setting in ['perm']:
            inputs = df.apply(lambda x: [
                [
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], permuted_options),
                ] for permuted_options in permute_options([x[e] for e in option_ids_header])
            ], axis=1).to_list()
        elif setting in ['cyclic']:
            inputs = df.apply(lambda x: [
                [
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], cycled_options),
                ] for cycled_options in cycle_options([x[e] for e in option_ids_header])
            ], axis=1).to_list()
        else:
            inputs = df.apply(lambda x: [
                sys_msg.format(subject.replace('_', ' ')),
                create_user_prompt(x["Question"], [x[e] for e in option_ids_header]),
            ], axis=1).to_list()
        options = df.apply(lambda x: [str(x[e]) for e in option_ids_header], axis=1).to_list()
        ideals = df.apply(lambda x: option_ids[option_ids_header.index(x["Answer"])], axis=1).to_list()
        return list(zip(inputs, options, ideals))

    # prepare_eval_fn
    if setting in ['noid']:
        prepare_eval_fn = partial(prepare_eval_fn_noid, num_few_shot=num_few_shot, option_ids=option_ids)
    elif setting in ['perm', 'cyclic']:
        prepare_eval_fn = partial(prepare_eval_fn_perm, num_few_shot=num_few_shot, option_ids=option_ids)
    else:
        prepare_eval_fn = partial(prepare_eval_fn_base, num_few_shot=num_few_shot, option_ids=option_ids)

    return subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn


def prepare_eval_fn_base(model, toker, few_shot_samples, num_few_shot, option_ids):
    bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]

    def eval_fn(sample, rng: random.Random):
        idx, (input, options, ideal) = sample
        sys_msg, eval_sample = input.copy()
        input_text = sys_msg + '\n\n'
        if num_few_shot > 0:
            for s in few_shot_samples[:num_few_shot]:
                input_text += s + '\n\n'
        input_text += eval_sample
        if not bpe_has_space_prefix:
            input_text += ' '

        input_ids = toker(input_text, return_tensors="pt").input_ids.to(model.device)
        input_ids = input_ids[..., -1536:]
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
            ).logits[:, -1].view(-1)

        option_indices = [toker(f': {e}').input_ids[-1] for e in option_ids] + \
            [toker(f':{e}').input_ids[-1] for e in option_ids]
        probs = F.softmax(
            logits[..., option_indices], dim=-1
        ).detach().cpu().to(torch.float32).numpy()
        probs = probs.reshape(2, len(option_ids)).sum(axis=0)
        sampled = option_ids[np.argmax(probs)]

        correct = (sampled == ideal)
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_text,
                'options': options,
                'probs': probs.tolist(),
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_noid(model, toker, few_shot_samples, num_few_shot, option_ids):
    toker.padding_side = 'right'
    bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]

    def eval_fn(sample, rng: random.Random):
        idx, (input, options, ideal) = sample
        sys_msg, eval_sample = input.copy()
        input_text = sys_msg + '\n\n'
        if num_few_shot > 0:
            for s in few_shot_samples[:num_few_shot]:
                input_text += s + '\n\n'
        input_text += eval_sample
        prefix_input_ids = toker(input_text, truncation=False, return_tensors="pt").input_ids

        losses = []
        lengths = []
        for option in options:
            prefix_and_option_text = input_text + ' ' + option.strip()
            input_ids = toker(prefix_and_option_text, truncation=False, return_tensors="pt").input_ids.to(model.device)
            lengths.append(input_ids.size(1) - prefix_input_ids.size(1))

            labels = input_ids.clone()
            labels[:, :prefix_input_ids.size(1)] = -100

            input_ids = input_ids[..., -1536:]
            labels = labels[..., -1536:]

            with torch.no_grad():
                loss = model(
                    input_ids=input_ids,
                    labels=labels,
                ).loss.detach().to(torch.float32).cpu().item()
            losses.append(loss)

        nll = - np.array(losses)
        probs = np.exp(nll - np.max(nll))
        probs = probs / (probs.sum() + 1e-10)

        sampled = option_ids[np.argmin(losses)]
        correct = (sampled == ideal)
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_text,
                'options': options,
                'lengths': lengths,
                'losses': losses,
                'probs': probs.tolist(),
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_perm(model, toker, few_shot_samples, num_few_shot, option_ids):
    toker.padding_side = 'left'
    bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]

    def eval_fn(sample, rng: random.Random):
        idx, (probing_inputs, options, ideal) = sample
        assert len(probing_inputs) in [4, 24, 5]

        input_texts = []
        for probing_input in probing_inputs:
            sys_msg, eval_sample = probing_input.copy()
            input_text = sys_msg + '\n\n'
            if num_few_shot > 0:
                for s in few_shot_samples[:num_few_shot]:
                    input_text += s + '\n\n'
            input_text += eval_sample
            if not bpe_has_space_prefix:
                input_text += ' '
            input_texts.append(input_text)

        all_probs = []
        for input_text in input_texts:
            input_ids = toker(input_text, truncation=False, return_tensors="pt").input_ids.to(model.device)
            input_ids = input_ids[..., -1536:]
            with torch.no_grad():
                logits = model(
                    input_ids=input_ids,
                ).logits[:, -1]

            option_indices = [toker(f': {e}').input_ids[-1] for e in option_ids] + \
                [toker(f':{e}').input_ids[-1] for e in option_ids]
            probs = F.softmax(
                logits[..., option_indices], dim=-1,
            ).detach().to(torch.float32).cpu().numpy()
            probs = probs.reshape(input_ids.size(0), 2, len(option_ids)).sum(axis=1)
            all_probs.extend(probs.tolist())

        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_texts[0],
                'options': options,
                'probs': all_probs,
                'ideal': ideal,
            },
        }
        return result
    return eval_fn

