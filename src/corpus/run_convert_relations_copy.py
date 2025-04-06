import json
import os
import pandas as pd

from src.corpus.data_process import get_train_data
from collections import defaultdict


if __name__ == '__main__':

    jsonl_path = '../../data/corpus/v2/admin-revisado.jsonl'

    relations_dir = '../../data/corpus/v2/relations_2/'

    os.makedirs(relations_dir, exist_ok=True)

    with open(file=jsonl_path, mode='r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]

    dataframe = pd.json_normalize(data)

    all_examples = get_train_data(dataframe)

    relations_count = defaultdict(dict)

    for example in all_examples:

        if example.relation in relations_count[example.id]:
            relations_count[example.id][example.relation] += 1
        else:
            relations_count[example.id][example.relation] = 1

        file_name = f'{example.relation}_{relations_count[example.id][example.relation]}.txt'

        directory = os.path.join(f'{relations_dir}', f'doc_{example.id}')

        if example.relation != 'O':
            directory = os.path.join(directory, 'positives')
        else:
            directory = os.path.join(directory, 'negatives')

        os.makedirs(directory, exist_ok=True)

        file_path = os.path.join(directory, file_name)

        text = example.text.replace('\n', ' ')

        with open(file=file_path, mode='w', encoding='utf-8') as file:
            file.write(f'{text}\t{example.relation}')

    unify_file_path = os.path.join(relations_dir, 'all_text_and_relation.txt')

    with open(file=unify_file_path, mode='w', encoding='utf-8') as file:

        for example in all_examples:
            text = example.text.replace('\n', ' ')
            data = f'{text}\t{example.relation}\n\n'
            file.write(data)
