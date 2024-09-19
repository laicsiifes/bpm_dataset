import os.path

from itertools import product


def extract_entities(text, list_entity_annotations, nlp):

    doc_text = nlp(text)

    sentences_spacy = [s for s in doc_text.sents]

    list_sentences = []

    for sentence in sentences_spacy:
        tokens = []
        for token in sentence:
            start_offset = token.idx
            end_offset = token.idx + len(token.text) - 1
            tokens.append([start_offset, end_offset, token.text])
        list_sentences.append((sentence.start_char, sentence.end_char, tokens))

    entities = []

    for entity_annotation in list_entity_annotations:

        id_ = entity_annotation['id']
        label = entity_annotation['label']
        start_entity = entity_annotation['start_offset']
        end_entity = entity_annotation['end_offset']
        entity = text[start_entity: end_entity].strip()

        """
            Os casos a seguir tiveram que ser corrigidos via código por problemas de divergência nas posições dos 
            tokens.
        """

        if 'As a basic principle,' in text:

            if start_entity == 2922:
                entity = text[start_entity - 1: end_entity].strip()

            if start_entity == 659 and end_entity == 660:
                entity = text[start_entity - 1: end_entity + 1].strip()

            if start_entity == 766 and end_entity == 768:
                entity = text[start_entity - 1: end_entity + 1].strip()

            if start_entity == 1532 and end_entity == 1535:
                entity = text[start_entity - 1: end_entity + 1].strip()

            if start_entity == 1582 and end_entity == 1588:
                start_entity = 1579
                entity = text[1579: end_entity + 1].strip()

            if start_entity == 1609 and end_entity == 1612:
                entity = text[start_entity - 1: end_entity + 1].strip()

            if start_entity == 1613 and end_entity == 1621:
                entity = text[start_entity - 1: end_entity + 1].strip()

            if start_entity == 1613 and end_entity == 1621:
                entity = text[start_entity - 1: end_entity + 1].strip()

            if start_entity == 1613 and end_entity == 1621:
                start_entity = 1611
                entity = text[start_entity - 1: end_entity + 1].strip()

            if start_entity == 1926 and end_entity == 1928:
                entity = text[start_entity: end_entity + 2].strip()

            if start_entity == 1992 and end_entity == 2030:
                start_entity = 1994
                entity = text[start_entity: end_entity].strip()

            if start_entity == 2036 and end_entity == 2041:
                entity = text[start_entity: end_entity + 3].strip()

            if start_entity == 2317 and end_entity == 2325:
                entity = text[start_entity: end_entity + 2].strip()

        if start_entity == 2922:
            entity = text[start_entity - 1: end_entity].strip()

        if start_entity == 658:
            entity = text[start_entity: end_entity + 1].strip()

        if start_entity == 776:
            entity = text[start_entity - 1: end_entity].strip()

        entities.append((id_, entity, label, start_entity, end_entity))

    return entities


def mark_entities(entity, label, text, window=5):

    if text.lower().count(entity[1].lower()) == 1:
        text = text.replace(entity[1], label)
        return text
    else:
        start_position = entity[3] - window
        end_position = entity[4] + window
        prefix_text = text[:start_position]
        fragment = text[start_position:end_position]
        fragment = fragment.replace(entity[1], label)
        sufix_text = text[end_position:]
        return f'{prefix_text}{fragment}{sufix_text}'


def save_relation_files(list_relation_annotations, list_entities, text, id_example,
                        relations_dir):

    dict_entities = {}

    for entity in list_entities:
        if entity[0] in dict_entities:
            print('ERRO')
            exit(-1)
        dict_entities[entity[0]] = entity

    dict_true_relations = {}

    entities_real_rel = []

    for relation_annotation in list_relation_annotations:

        relation_type = relation_annotation['type']

        entity_from = dict_entities[relation_annotation['from_id']]
        entity_to = dict_entities[relation_annotation['to_id']]

        label_1 = f'[E1]{entity_from[1]}[/E1]'
        label_2 = f'[E2]{entity_to[1]}[/E2]'

        label_entities = f'{entity_from[1]}___{entity_to[1]}'

        entities_real_rel.append(label_entities)

        marked_text = mark_entities(entity_from, label_1, text, window=5)

        marked_text = mark_entities(entity_to, label_2, marked_text, window=14)

        marked_text = marked_text.replace('\n', '').strip()

        if relation_type not in dict_true_relations:
            dict_true_relations[relation_type] = []

        dict_true_relations[relation_type].append(marked_text)

    example_dir = os.path.join(relations_dir, str(id_example))

    os.makedirs(example_dir, exist_ok=True)

    positive_examples_dir = os.path.join(example_dir, 'positive')

    os.makedirs(positive_examples_dir, exist_ok=True)

    for relation_type, list_marked_texts in dict_true_relations.items():

        for i, marked_text in enumerate(list_marked_texts, start=1):

            file_path = os.path.join(positive_examples_dir, f'{relation_type}__{i}.txt')

            with open(file=file_path, mode='w', encoding='utf-8') as file:
                file.write(marked_text)

    negative_examples_dir = os.path.join(example_dir, 'negative')

    os.makedirs(negative_examples_dir, exist_ok=True)

    entities_real_rel = set(entities_real_rel)

    set_entities = set(list_entities)

    list_false_relations = []

    for entity_from, entity_to in product(set_entities, set_entities):
        if entity_from[1] != entity_to[1]:
            label_entities = f'{entity_from[1]}___{entity_to[1]}'
            inver_label_entities = f'{entity_to[1]}___{entity_from[1]}'
            if label_entities not in entities_real_rel and \
                    inver_label_entities not in entities_real_rel:
                list_false_relations.append((entity_from, entity_to))

    for i, false_relation in enumerate(list_false_relations, start=1):

        entity_from = false_relation[0]
        entity_to = false_relation[1]

        label_1 = f'[E1]{entity_from[1]}[/E1]'
        label_2 = f'[E2]{entity_to[1]}[/E2]'

        marked_text = mark_entities(entity_from, label_1, text, window=5)

        marked_text = mark_entities(entity_to, label_2, marked_text, window=14)

        marked_text = marked_text.replace('\n', '').strip()

        file_path = os.path.join(negative_examples_dir, f'O__{i}.txt')

        with open(file=file_path, mode='w', encoding='utf-8') as file:
            file.write(marked_text)


