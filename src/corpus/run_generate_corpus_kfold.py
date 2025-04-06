import json
import os
import random
import numpy as np
import spacy

from src.corpus.data_utils import extract_sentences_entities, extract_entities, extract_relations
from src.corpus.corpus_utils import save_conll_file, save_relations_csv


if __name__ == '__main__':

    jsonl_path = '../../data/corpus/v2/admin-revisado.jsonl'

    folders_dir = '../../data/corpus/v2/folders'

    nlp = spacy.load('en_core_web_sm')

    os.makedirs(folders_dir, exist_ok=True)

    with open(file=jsonl_path, mode='r', encoding='utf-8') as file:
        list_examples = [json.loads(line.strip()) for line in file]

    print(f'\nTotal Examples: {len(list_examples)}')

    list_ids = list(range(len(list_examples)))

    random.shuffle(list_ids)

    chunks = np.array_split(list_ids, indices_or_sections=5)

    folders_all_entities = []
    folders_all_text_marked_relations = []
    folders_all_label_relations = []

    for k, chunk in enumerate(chunks, start=1):

        print(f'\nFolder {k} -- {chunk}')

        list_examples_folder = [list_examples[id_] for id_ in range(len(list_examples)) if id_ in chunk]

        all_sentences = []

        all_text_marked = []
        all_label_relations = []

        for example in list_examples_folder:

            list_sentences = extract_sentences_entities(example, nlp)

            all_sentences.extend(list_sentences)

            text = example['text']
            list_entity_annotations = example['entities']
            list_relation_annotations = example['relations']

            entities = extract_entities(text, list_entity_annotations, nlp)

            list_text_marked, list_label_relations = extract_relations(
                text, list_relation_annotations, entities)

            all_text_marked.extend(list_text_marked)
            all_label_relations.extend(list_label_relations)

        folders_all_entities.append(all_sentences)
        folders_all_text_marked_relations.append(all_text_marked)
        folders_all_label_relations.append(all_label_relations)

    for test_k in range(len(folders_all_entities)):

        test_sentences_entities = folders_all_entities[test_k]

        test_texts_relations = folders_all_text_marked_relations[test_k]
        test_labels_relations = folders_all_label_relations[test_k]

        val_k = (test_k + 1) % len(folders_all_entities)

        val_sentences_entities = folders_all_entities[val_k]

        validation_texts_relations = folders_all_text_marked_relations[val_k]
        validation_labels_relations = folders_all_label_relations[val_k]

        train_sentences_entities = []
        train_texts_relations = []
        train_labels_relations = []

        for train_k in range(len(folders_all_entities)):

            if train_k != test_k and train_k != val_k:

                train_sentences_entities.extend(folders_all_entities[train_k])

                train_texts_relations.extend(folders_all_text_marked_relations[train_k])

                train_labels_relations.extend(folders_all_label_relations[train_k])

        folder_k_dir = os.path.join(folders_dir, f'{test_k + 1}')

        os.makedirs(folder_k_dir, exist_ok=True)

        train_conll_file_path = os.path.join(folder_k_dir, f'train_{test_k + 1}.conll')
        val_conll_file_path = os.path.join(folder_k_dir, f'val_{test_k + 1}.conll')
        test_conll_file_path = os.path.join(folder_k_dir, f'test_{test_k + 1}.conll')

        save_conll_file(train_sentences_entities, train_conll_file_path)
        save_conll_file(val_sentences_entities, val_conll_file_path)
        save_conll_file(test_sentences_entities, test_conll_file_path)

        train_csv_file_path = os.path.join(folder_k_dir, f'train_rel_{test_k + 1}.csv')
        val_csv_file_path = os.path.join(folder_k_dir, f'val_rel_{test_k + 1}.csv')
        test_csv_file_path = os.path.join(folder_k_dir, f'test_rel_{test_k + 1}.csv')

        save_relations_csv(train_texts_relations, train_labels_relations, train_csv_file_path)
        save_relations_csv(validation_texts_relations, validation_labels_relations, val_csv_file_path)
        save_relations_csv(test_texts_relations, test_labels_relations, test_csv_file_path)
