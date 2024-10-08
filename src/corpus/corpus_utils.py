import os
import pandas as pd


def save_conll_file(list_sentences: list, conll_file_path: str):
    conll_file = ''
    for sentence in list_sentences:
        tokens = [s[2] for s in sentence[2]]
        labels = [s[3] for s in sentence[2]]
        # print(f'\nTokens: {tokens}')
        # print(f'Labels: {labels}')
        for token, label in zip(tokens, labels):
            if '\n' not in token:
                conll_file += f'{token}\t{label}\n'
        conll_file += '\n'
    with open(file=conll_file_path, mode='w', encoding='utf-8') as file:
        file.write(conll_file)


def read_corpus_file(corpus_file: str, delimiter: str = '\t', ner_column: int = 1) -> list:
    with open(corpus_file, encoding='utf-8') as file:
        lines = file.readlines()
    data = []
    words = []
    tags = []
    for line in lines:
        line = line.replace('\n', '')
        if line != '':
            if delimiter in line:
                fragments = line.split(delimiter)
                words.append(fragments[0])
                tags.append(fragments[ner_column])
        else:
            if len(words) > 1:
                data.append((words, tags))
            words = []
            tags = []
    return data


def get_examples(examples_folder: str, limit: int = -1) -> tuple[list[str], list[str]]:
    list_example_names = os.listdir(examples_folder)
    list_texts = []
    list_labels = []
    for example_name in list_example_names:
        example_path = os.path.join(examples_folder, example_name)
        if example_path.endswith('.txt'):
            with open(file=example_path, mode='r', encoding='utf-8') as file:
                content = file.read()
            label = example_name.split('__')[0]
            list_texts.append(content)
            list_labels.append(label)
            if len(list_texts) == limit:
                break
    return list_texts, list_labels




def read_corpus_relations(corpus_folder: str):
    list_docs_names = os.listdir(corpus_folder)
    corpus_texts = []
    corpus_labels = []
    for doc_name in list_docs_names:
        doc_path = os.path.join(corpus_folder, doc_name)
        positive_folder = os.path.join(doc_path, 'positive')
        negative_folder = os.path.join(doc_path, 'negative')
        positive_texts, positive_labels = get_examples(positive_folder)
        negative_texts, negative_labels = get_examples(negative_folder,
                                                       limit=len(positive_texts))
        all_texts = []
        all_labels = []
        all_texts.extend(positive_texts)
        all_texts.extend(negative_texts)
        all_labels.extend(positive_labels)
        all_labels.extend(negative_labels)
        corpus_texts.append(all_texts)
        corpus_labels.append(all_labels)
    return corpus_texts, corpus_labels


def save_corpus_csv(list_documents, list_labels, list_idx, csv_file_path):

    all_examples = []
    all_labels = []

    for id_ in list_idx:
        examples = list_documents[id_]
        labels = list_labels[id_]
        all_examples.extend(examples)
        all_labels.extend(labels)

    dataframe = pd.DataFrame(
        {
            'texts': all_examples,
            'labels': all_labels
        },
        columns=['texts', 'labels']
    )

    dataframe.to_csv(csv_file_path, mode='w', index=False, encoding='utf-8')


def save_relations_csv(list_texts, list_labels, csv_file_path):

    dataframe = pd.DataFrame(
        {
            'texts': list_texts,
            'labels': list_labels
        },
        columns=['texts', 'labels']
    )

    dataframe.to_csv(csv_file_path, mode='w', index=False, encoding='utf-8')


def read_relations_csv(csv_file_path: str):

    dataframe = pd.read_csv(csv_file_path, delimiter=',', encoding='utf-8')

    texts = dataframe['texts'].values
    labels = dataframe['labels'].values

    return texts, labels
