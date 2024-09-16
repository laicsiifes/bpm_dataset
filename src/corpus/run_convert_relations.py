import json
import os
import spacy

from src.corpus.data_utils import extract_entities, save_relation_files


if __name__ == '__main__':

    jsonl_path = '../../data/corpus/anotacoes/admin.jsonl'

    relations_dir = '../../data/corpus/relations/'

    nlp = spacy.load('en_core_web_sm')

    os.makedirs(relations_dir, exist_ok=True)

    with open(file=jsonl_path, mode='r', encoding='utf-8') as file:
        list_examples = [json.loads(line.strip()) for line in file]

    for example in list_examples:

        id_example = example['id']
        text = example['text']

        text_aux = text.replace('\n', ' ')

        print(f'\nDocument: {id_example} -- {text_aux}')

        list_entity_annotations = example['entities']
        list_relation_annotations = example['relations']

        entities = extract_entities(text, list_entity_annotations, nlp)

        print(f'\n  Total Entities: {len(list_entity_annotations)} -- {len(entities)}')

        print(f'\n  Total Relations: {len(list_relation_annotations)}')

        save_relation_files(list_relation_annotations, entities, text, id_example,
                            relations_dir)

    print(f'\n\nTotal Examples: {len(list_examples)}')
