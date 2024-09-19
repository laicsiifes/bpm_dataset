import random

from src.corpus.corpus_utils import read_corpus_relations, save_corpus_csv


if __name__ == '__main__':

    relations_dir = '../../data/corpus/relations/'

    documents, labels = read_corpus_relations(relations_dir)

    print(f'\nTotal: {len(documents)} -- {len(labels)}')

    list_docs_idx = list(range(len(documents)))

    n_examples_test = round(len(list_docs_idx) * 0.10)
    n_examples_val = round(len(list_docs_idx) * 0.10)
    n_examples_train = len(list_docs_idx) - n_examples_test - n_examples_val

    print(f'\nTrain: {n_examples_train}')
    print(f'Val: {n_examples_val}')
    print(f'Test: {n_examples_test}')

    list_test_idx = random.choices(list_docs_idx, k=n_examples_test)

    list_docs_idx = [id_ for id_ in list_docs_idx if id_ not in list_test_idx]

    list_val_idx = random.choices(list_docs_idx, k=n_examples_val)

    list_train_idx = [id_ for id_ in list_docs_idx if id_ not in list_val_idx]

    list_train_idx.sort()
    list_val_idx.sort()
    list_test_idx.sort()

    print(f'\nTrain: {list_train_idx}')
    print(f'Validation: {list_val_idx}')
    print(f'Test: {list_test_idx}')

    list_sentences_idx = list(range(len(documents)))

    train_file_path = '../../data/corpus/train_rel.csv'
    val_file_path = '../../data/corpus/validation_rel.csv'
    test_file_path = '../../data/corpus/test_rel.csv'

    save_corpus_csv(documents, labels, list_train_idx, train_file_path)

    save_corpus_csv(documents, labels, list_val_idx, val_file_path)

    save_corpus_csv(documents, labels, list_test_idx, test_file_path)
