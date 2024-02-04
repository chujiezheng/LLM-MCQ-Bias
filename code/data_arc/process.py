
import json
import os
import sys
import random
import csv

for key in ['dev', 'test']:
    data = [json.loads(e) for e in open(f'ARC-Challenge-{key.capitalize()}.jsonl')]
    if key == 'dev':
        data = data[:5]

    cnt = 0
    writer = csv.writer(open(f'{key}/arc_{key}.csv', 'w'))
    rdx = 0
    for row in data:
        question = row['question']['stem']
        choices = {e['label']: e['text'] for e in row['question']['choices']}
        if len(choices) != 4:
            cnt += 1
            continue
        choices = [choices[e] for e in sorted(choices.keys())]
        label = row['answerKey']
        if label not in ['A', 'B', 'C', 'D']:
            assert int(label) in [1, 2, 3, 4], label
            label = 'ABCD'[int(label) - 1]
        rdx += 1

        writer.writerow([question, choices[0], choices[1], choices[2], choices[3], label])

    print(cnt)
    