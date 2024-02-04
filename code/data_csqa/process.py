
import json
import os
import sys
import random
import csv

for key in ['dev', 'test']:
    writer = csv.writer(open(f'{key}/csqa_{key}.csv', 'w'))

    data = [json.loads(e) for e in open(f'dev_rand_split.jsonl')]
    random.Random(23).shuffle(data)
    if key == 'dev':
        data = data[:5]
    else:
        data = data[5:]

    for row in data:
        question = row['question']['stem']
        choices = {e['label']: e['text'] for e in row['question']['choices']}
        if len(choices) != 5:
            continue
        choices = [choices[e] for e in sorted(choices.keys())]
        label = row['answerKey']
        if label not in ['A', 'B', 'C', 'D', 'E']:
            raise ValueError

        writer.writerow([question, choices[0], choices[1], choices[2], choices[3], choices[4], label])
