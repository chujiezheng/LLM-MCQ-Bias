
import os
import sys
import random
import json
import argparse
import copy
import logging
from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
from utils import (
    _norm,
    shuffle_options_with_ids,
    move_answer,
    cycle_options,
    permute_options,
)
from functools import partial

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_names", type=str, nargs='+', default=[],
                        help='eval tasks and settings')
    parser.add_argument("--mini", action='store_true')
    args = parser.parse_args()

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
                'noid', 'cot', 'shuffle_both',
                'cyclic',
            ] or (setting.startswith('move') and setting[-1] in ['a', 'b', 'c', 'd'])
        ):
            raise ValueError(f"Unknown setting: {setting}")
        if setting in ['cot']:
            assert num_few_shot == 0

    return args


def prepare_eval(args, eval_name):
    # task and setting
    eval_args = eval_name.split(',')
    args.task = task = eval_args[0]
    if 'mmlu' in task:
        args.max_num_samples = 100
    else:
        args.max_num_samples = None
    args.num_few_shot = num_few_shot = int(eval_args[1])
    args.setting = setting = eval_args[2] if len(eval_args) > 2 else None
    if setting is not None and setting.startswith('move'):
        moved_answer = setting[-1].upper()

    # save_path
    save_path = f'results_{task}/{num_few_shot}s_ichat/{task}'
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

    if setting in ['cot']:
        sys_msg += ' You should reason in a step-by-step manner as to get the right answer.'
        cot_msgs = [
            "Let's think step by step:",
            "\nGiven all of the above, the answer of the question is:",
        ]
    else:
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
            few_shot_samples = df.apply(lambda x: [
                {"role": "user", "content": create_user_prompt(x["Question"], [x[e] for e in option_ids_header])},
                {"role": "assistant", "content": str(x[x["Answer"]])},
            ], axis=1).to_list()
        else:
            few_shot_samples = df.apply(lambda x: [
                {"role": "user", "content": create_user_prompt(x["Question"], [x[e] for e in option_ids_header])},
                {"role": "assistant", "content": option_ids[option_ids_header.index(x["Answer"])]},
            ], axis=1).to_list()
        return few_shot_samples

    # prepare_eval_samples
    def prepare_eval_samples(subject):
        df = pd.read_csv(open(f'{data_path}/test/{subject}_test.csv'), names=("Question", *option_ids_header, "Answer"), dtype=str)

        if setting is not None and setting.startswith('move'):
            df = df.apply(lambda x: move_answer(x, moved_answer), axis=1)

        if setting in ['cyclic']:
            inputs = df.apply(lambda x: [
                [
                    {"role": "system", "content": sys_msg.format(subject.replace('_', ' '))},
                    {"role": "user", "content": create_user_prompt(x["Question"], cycled_options)},
                ] for cycled_options in cycle_options([x[e] for e in option_ids_header])
            ], axis=1).to_list()
        else:
            inputs = df.apply(lambda x: [
                {"role": "system", "content": sys_msg.format(subject.replace('_', ' '))},
                {"role": "user", "content": create_user_prompt(x["Question"], [x[e] for e in option_ids_header])},
            ], axis=1).to_list()
        options = df.apply(lambda x: [str(x[e]) for e in option_ids_header], axis=1).to_list()
        ideals = df.apply(lambda x: option_ids[option_ids_header.index(x["Answer"])], axis=1).to_list()
        return list(zip(inputs, options, ideals))

    # prepare_eval_fn
    if setting in ['cot']:
        prepare_eval_fn = partial(prepare_eval_fn_cot, num_few_shot=num_few_shot, cot_msgs=cot_msgs)
    elif setting in ['noid']:
        prepare_eval_fn = partial(prepare_eval_fn_noid, num_few_shot=num_few_shot, option_ids=option_ids)
    elif setting in ['cyclic']:
        prepare_eval_fn = partial(prepare_eval_fn_perm, num_few_shot=num_few_shot, option_ids=option_ids)
    else:
        prepare_eval_fn = partial(prepare_eval_fn_base, num_few_shot=num_few_shot)

    return subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn


def prepare_eval_fn_base(api, few_shot_samples, num_few_shot):
    def eval_fn(sample, rng: random.Random):
        idx, (input, options, ideal) = sample
        messages = input[:-1]
        for s in few_shot_samples[:num_few_shot]:
            messages += copy.deepcopy(s)
        messages += input[-1:]

        result = api(model='gpt-3.5-turbo', messages=messages, temperature=0.0, max_tokens=2)
        if result is None or 'content' not in result['choices'][0]['message']:
            return {
                'type': 'broken',
                'data': {
                    'idx': idx,
                    'prompt': messages,
                    'options': options,
                    'ideal': ideal,
                },
            }

        sampled = result['choices'][0]['message']['content']
        correct = _norm(sampled).startswith(_norm(ideal))
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': messages,
                'options': options,
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_noid(api, few_shot_samples, num_few_shot, option_ids):
    def eval_fn(sample, rng: random.Random):
        idx, (input, options, ideal) = sample
        messages = input[:-1]
        for s in few_shot_samples[:num_few_shot]:
            messages += copy.deepcopy(s)
        messages += input[-1:]

        result = api(model='gpt-3.5-turbo', messages=messages, temperature=0.0, max_tokens=0)
        if result is None or 'content' not in result['choices'][0]['message']:
            return {
                'type': 'broken',
                'data': {
                    'idx': idx,
                    'prompt': messages,
                    'options': options,
                    'ideal': ideal,
                },
            }

        sampled = result['choices'][0]['message']['content']
        correct = _norm(sampled).startswith(_norm(options[option_ids.index(ideal)]))
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': messages,
                'options': options,
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_perm(api, few_shot_samples, num_few_shot, option_ids):
    probing_temp = 1.0
    probing_p = 0.99
    probing_n = 100

    def eval_fn(sample, rng: random.Random):
        idx, (probing_inputs, options, ideal) = sample
        assert len(probing_inputs) in [4, 5]

        observed = []
        all_messages = []
        for probing_input in probing_inputs:
            sys_msg, eval_sample = probing_input
            messages = [sys_msg]
            for s in few_shot_samples[:num_few_shot]:
                messages += s
            messages += [eval_sample]
            all_messages.append(messages)

            result = api(model='gpt-3.5-turbo', messages=messages, temperature=probing_temp, top_p=probing_p, n=probing_n, max_tokens=2)
            if result is None:
                return {
                    'type': 'broken',
                    'data': {
                        'idx': idx,
                        'prompt': probing_input,
                        'options': options,
                        'ideal': ideal,
                    },
                }

            sampled = [choice['message']['content'] if 'content' in choice['message'] else '' for choice in result['choices']]
            obs = [sum([1 for e in sampled if _norm(e).startswith(option_id)]) for option_id in option_ids]
            observed.append(obs)

        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': all_messages[0],
                'options': options,
                'observed': observed,
                'ideal': ideal,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_cot(api, few_shot_samples, num_few_shot, cot_msgs):
    def eval_fn(sample, rng: random.Random):
        idx, (input, options, ideal) = sample
        messages = input[:-1]
        for s in few_shot_samples[:num_few_shot]:
            messages += copy.deepcopy(s)
        messages += input[-1:]

        messages += [{'content': cot_msgs[0], 'role': 'assistant',}]
        result = api(model='gpt-3.5-turbo', messages=messages, temperature=0.0, max_tokens=0)
        if result is None or 'content' not in result['choices'][0]['message']:
            return {
                'type': 'broken',
                'data': {
                    'idx': idx,
                    'prompt': messages,
                    'options': options,
                    'ideal': ideal,
                },
            }
        cot = result['choices'][0]['message']['content']
        messages += [{'content': cot, 'role': 'assistant',}]
        messages += [{'content': cot_msgs[1], 'role': 'assistant',}]
        result = api(model='gpt-3.5-turbo', messages=messages, temperature=0.0, max_tokens=2)
        if result is None or 'content' not in result['choices'][0]['message']:
            return {
                'type': 'broken',
                'data': {
                    'idx': idx,
                    'prompt': messages,
                    'options': options,
                    'ideal': ideal,
                },
            }

        sampled = result['choices'][0]['message']['content']
        correct = _norm(sampled).startswith(_norm(ideal))
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': messages,
                'options': options,
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn

