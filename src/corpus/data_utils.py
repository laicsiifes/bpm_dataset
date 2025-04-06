from itertools import product


def extract_sentences_entities(example, nlp):

    text = example['text']

    list_entity_annotations = example['entities']

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

    cont = 0

    for entity_annotation in list_entity_annotations:

        label = entity_annotation['label']
        start_entity = entity_annotation['start_offset']
        end_entity = entity_annotation['end_offset']

        entity = text[start_entity: end_entity].strip()

        if start_entity == 1188 and entity == 'custome':
            entity = text[start_entity: end_entity + 1].strip()
        elif start_entity == 5486 and entity == 'n the case of private insurance':
            entity = text[start_entity - 1: end_entity + 1].strip()
        elif start_entity == 315 and entity == 'he application is returned to the applicant/employee':
            entity = text[start_entity - 1: end_entity + 1].strip()
        elif start_entity == 80 and entity == 'pplications are recorded':
            entity = text[start_entity - 1: end_entity + 1].strip()
        elif start_entity == 776 and entity == 'eviews the dossier':
            entity = text[start_entity - 1: end_entity + 1].strip()

        is_founded = False

        first_token = entity.split()[0]

        for sentence in list_sentences:
            if sentence[0] <= end_entity <= sentence[1]:
                for token in sentence[2]:
                    is_position_right = (token[0] == start_entity or (token[0] - 1) ==
                                         start_entity or (token[0] + 1) == start_entity)
                    if is_position_right and token[2] in first_token:
                        token.append(f'B-{label}')
                        is_founded = True
                        cont += 1
                    elif start_entity < token[0] < end_entity:
                        token.append(f'I-{label}')
                break

        if not is_founded:
            print('\t\t====> ERROR.')
            exit(-1)

    for sentence in list_sentences:
        for token in sentence[2]:
            if len(token) == 3:
                token.append('O')

    return list_sentences


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

        if start_entity == 1188 and entity == 'custome':
            entity = text[start_entity: end_entity + 1].strip()
        elif start_entity == 5486 and entity == 'n the case of private insurance':
            entity = text[start_entity - 1: end_entity + 1].strip()
        elif start_entity == 315 and entity == 'he application is returned to the applicant/employee':
            entity = text[start_entity - 1: end_entity + 1].strip()
        elif start_entity == 80 and entity == 'pplications are recorded':
            entity = text[start_entity - 1: end_entity + 1].strip()
        elif start_entity == 776 and entity == 'eviews the dossier':
            entity = text[start_entity - 1: end_entity + 1].strip()

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


def extract_relations(text, list_relation_annotations, list_entities):

    dict_entities = {}

    for entity in list_entities:
        if entity[0] in dict_entities:
            print('ERRO')
            exit(-1)
        dict_entities[entity[0]] = entity

    entities_real_rel = []

    all_texts = []
    all_labels = []

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

        all_texts.append(marked_text)
        all_labels.append(relation_type)

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

    double_total_positive = 2 * len(all_texts)

    for i, false_relation in enumerate(list_false_relations, start=1):

        entity_from = false_relation[0]
        entity_to = false_relation[1]

        label_1 = f'[E1]{entity_from[1]}[/E1]'
        label_2 = f'[E2]{entity_to[1]}[/E2]'

        marked_text = mark_entities(entity_from, label_1, text, window=5)

        marked_text = mark_entities(entity_to, label_2, marked_text, window=14)

        marked_text = marked_text.replace('\n', '').strip()

        all_texts.append(marked_text)
        all_labels.append('O')

        if len(all_texts) > double_total_positive:
            break

    return all_texts, all_labels


# def save_relation_files(list_relation_annotations, list_entities, text, id_example,
#                         relations_dir):
#
#     dict_entities = {}
#
#     for entity in list_entities:
#         if entity[0] in dict_entities:
#             print('ERRO')
#             exit(-1)
#         dict_entities[entity[0]] = entity
#
#     dict_true_relations = {}
#
#     entities_real_rel = []
#
#     for relation_annotation in list_relation_annotations:
#
#         relation_type = relation_annotation['type']
#
#         entity_from = dict_entities[relation_annotation['from_id']]
#         entity_to = dict_entities[relation_annotation['to_id']]
#
#         label_1 = f'[E1]{entity_from[1]}[/E1]'
#         label_2 = f'[E2]{entity_to[1]}[/E2]'
#
#         label_entities = f'{entity_from[1]}___{entity_to[1]}'
#
#         entities_real_rel.append(label_entities)
#
#         marked_text = mark_entities(entity_from, label_1, text, window=5)
#
#         marked_text = mark_entities(entity_to, label_2, marked_text, window=14)
#
#         marked_text = marked_text.replace('\n', '').strip()
#
#         if relation_type not in dict_true_relations:
#             dict_true_relations[relation_type] = []
#
#         dict_true_relations[relation_type].append(marked_text)
#
#     example_dir = os.path.join(relations_dir, str(id_example))
#
#     os.makedirs(example_dir, exist_ok=True)
#
#     positive_examples_dir = os.path.join(example_dir, 'positive')
#
#     os.makedirs(positive_examples_dir, exist_ok=True)
#
#     for relation_type, list_marked_texts in dict_true_relations.items():
#
#         for i, marked_text in enumerate(list_marked_texts, start=1):
#
#             file_path = os.path.join(positive_examples_dir, f'{relation_type}__{i}.txt')
#
#             with open(file=file_path, mode='w', encoding='utf-8') as file:
#                 file.write(marked_text)
#
#     negative_examples_dir = os.path.join(example_dir, 'negative')
#
#     os.makedirs(negative_examples_dir, exist_ok=True)
#
#     entities_real_rel = set(entities_real_rel)
#
#     set_entities = set(list_entities)
#
#     list_false_relations = []
#
#     for entity_from, entity_to in product(set_entities, set_entities):
#         if entity_from[1] != entity_to[1]:
#             label_entities = f'{entity_from[1]}___{entity_to[1]}'
#             inver_label_entities = f'{entity_to[1]}___{entity_from[1]}'
#             if label_entities not in entities_real_rel and \
#                     inver_label_entities not in entities_real_rel:
#                 list_false_relations.append((entity_from, entity_to))
#
#     for i, false_relation in enumerate(list_false_relations, start=1):
#
#         entity_from = false_relation[0]
#         entity_to = false_relation[1]
#
#         label_1 = f'[E1]{entity_from[1]}[/E1]'
#         label_2 = f'[E2]{entity_to[1]}[/E2]'
#
#         marked_text = mark_entities(entity_from, label_1, text, window=5)
#
#         marked_text = mark_entities(entity_to, label_2, marked_text, window=14)
#
#         marked_text = marked_text.replace('\n', '').strip()
#
#         file_path = os.path.join(negative_examples_dir, f'O__{i}.txt')
#
#         with open(file=file_path, mode='w', encoding='utf-8') as file:
#             file.write(marked_text)
#
#
