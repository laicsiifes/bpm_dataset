import os
import pandas as pd
import spacy
import random

from src.corpus.corpus_utils import save_conll_file


if __name__ == '__main__':

    file_path = '../../data/corpus/anotacoes/admin.jsonl'

    conll_file_path = '../../data/corpus/conll/'

    os.makedirs(conll_file_path, exist_ok=True)

    nlp = spacy.load('en_core_web_sm')

    json_dataframe = pd.read_json(path_or_buf=file_path, lines=True)

    all_sentences = []

    print('\nConverting to CONLL')

    for index, row in json_dataframe.iterrows():

        text = row['text']

        text_aux = text.replace('\n', ' ')

        print(f'\nText: {text_aux}')

        list_entity_annotations = row['entities']

        doc_text = nlp(text)

        sentences_spacy = [s for s in doc_text.sents]

        list_sentences = []

        for sentence in sentences_spacy:
            sentence_aux = sentence.text.replace('\n', '')
            tokens = []
            for token in sentence:
                start_offset = token.idx
                end_offset = token.idx + len(token.text) - 1
                tokens.append([start_offset, end_offset, token.text])
            list_sentences.append((sentence.start_char, sentence.end_char, tokens))
            print(f'\tSentence: {sentence.start_char} - {sentence.end_char} - {tokens}')

        print('\n')

        all_entities = []

        cont = 0

        for entity_annotation in list_entity_annotations:

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

            print(f'\tEntity Annotation: {label} - {entity} - {start_entity} - {end_entity}')

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
                        print(f'\t\tToken: {token}')
                    break

            if not is_founded:
                print('\t\t====> ERROR.')
                exit(-1)

        print(len(list_entity_annotations), '--', cont, '\n')

        for sentence in list_sentences:
            for token in sentence[2]:
                if len(token) == 3:
                    token.append('O')
            print(sentence)

        all_sentences.append(list_sentences)

    print(f'\nTotal of Documents: {len(all_sentences)}')

    list_sentences_idx = list(range(len(all_sentences)))

    n_examples_test = round(len(list_sentences_idx) * 0.10)
    n_examples_val = round(len(list_sentences_idx) * 0.10)
    n_examples_train = len(list_sentences_idx) - n_examples_test - n_examples_val

    print(f'\nTrain: {n_examples_train}')
    print(f'Val: {n_examples_val}')
    print(f'Test: {n_examples_test}')
    #
    # list_test_idx = random.choices(list_sentences_idx, k=n_examples_test)
    #
    # list_sentences_idx = [idx for idx in list_sentences_idx if idx not in list_test_idx]
    #
    # list_val_idx = random.choices(list_sentences_idx, k=n_examples_val)
    #
    # list_train_idx = [idx for idx in list_sentences_idx if idx not in list_val_idx]
    #
    # list_train_idx.sort()
    # list_val_idx.sort()
    # list_test_idx.sort()
    #
    # print(f'\nTrain: {list_train_idx}')
    # print(f'Validation: {list_val_idx}')
    # print(f'Test: {list_test_idx}')
    #
    # list_sentences_idx = list(range(len(all_sentences)))
    #
    # train_sentences = []
    # val_sentences = []
    # test_sentences = []
    #
    # for idx in list_sentences_idx:
    #     if idx in list_train_idx:
    #         train_sentences.extend(all_sentences[idx])
    #     elif idx in list_val_idx:
    #         val_sentences.extend(all_sentences[idx])
    #     elif idx in list_test_idx:
    #         test_sentences.extend(all_sentences[idx])
    #
    # print(f'\nTrain: {len(train_sentences)}')
    # print(f'Val: {len(val_sentences)}')
    # print(f'Test: {len(test_sentences)}')
    #
    # train_conll_file_path = os.path.join(conll_file_path, 'train.conll')
    # val_conll_file_path = os.path.join(conll_file_path, 'validation.conll')
    # test_conll_file_path = os.path.join(conll_file_path, 'test.conll')
    #
    # save_conll_file(train_sentences, train_conll_file_path)
    # save_conll_file(val_sentences, val_conll_file_path)
    # save_conll_file(test_sentences, test_conll_file_path)
