import os
import matplotlib.pyplot as plt
import optuna

from src.corpus.corpus_utils import read_corpus_file
from src.ner.ner_features import data_preprocessing, convert_data
from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report
from src.ner.ner_utils import dump_report
from sklearn.metrics import ConfusionMatrixDisplay


def objective(trial):
    params = {
        'c1': trial.suggest_float('c1', low=0, high=2, step=0.2),
        'c2': trial.suggest_float('c2', low=0, high=2, step=0.2),
        'max_iterations': trial.suggest_int('max_iterations', low=0, high=2000, step=100),
    }
    crf_ = CRF(**params, algorithm='lbfgs', all_possible_transitions=True)
    crf_.fit(X_train, y_train)
    y_pred_ = crf_.predict(X_val)
    dict_report_ = classification_report(y_val, y_pred_, output_dict=True)
    return dict_report_['micro avg']['f1-score']


if __name__ == '__main__':

    delimiter = '\t'

    ner_column = 1

    report_dir = '../../data/results/ner/crf'

    train_file = '../../data/corpus/v2/train.conll'
    validation_file = '../../data/corpus/v2/validation.conll'
    test_file = '../../data/corpus/v2/test.conll'

    os.makedirs(report_dir, exist_ok=True)

    report_file = os.path.join(report_dir, 'results_crf.csv')

    train_data = read_corpus_file(train_file, delimiter=delimiter, ner_column=ner_column)
    validation_data = read_corpus_file(validation_file, delimiter=delimiter, ner_column=ner_column)
    test_data = read_corpus_file(test_file, delimiter=delimiter, ner_column=ner_column)

    print(f'\n  Train data: {len(train_data)}')
    print(f'  Validation data: {len(validation_data)}')
    print(f'  Test data: {len(test_data)}')

    print('\nPreprocessing ...')

    print('\n  Train data')

    train_data = data_preprocessing(train_data)

    print('  Validation data')

    validation_data = data_preprocessing(validation_data)

    print('  Test data')

    test_data = data_preprocessing(test_data)

    X_train, y_train = convert_data(train_data)
    X_val, y_val = convert_data(validation_data)
    X_test, y_test = convert_data(test_data)

    print(f'\nExample features: {X_train[-1]}')
    print(f'Tags: {y_train[-1]}')

    all_train_labels = []

    for y in y_train:
        all_train_labels.extend(y)

    all_train_labels = set(all_train_labels)

    all_validation_labels = []

    for y in y_val:
        all_validation_labels.extend(y)

    all_validation_labels = set(all_validation_labels)

    all_test_labels = []

    for y in y_test:
        all_test_labels.extend(y)

    all_test_labels = set(all_test_labels)

    print(f'\nTrain Labels: {len(all_train_labels)} -- {all_train_labels}')
    print(f'Validation Labels: {len(all_validation_labels)} -- {all_validation_labels}')
    print(f'Test Labels: {len(all_test_labels)} -- {all_test_labels}')

    study = optuna.create_study(direction='maximize')

    study.optimize(objective, n_trials=10)

    print('\nBest trial: ')

    best_trial = study.best_trial

    print(f'\n\tValue: {best_trial.value:.3f}')

    print('\tParams: ')

    for key, value in best_trial.params.items():
        print(f'\t  {key}: {value}')

    crf = CRF(**best_trial.params, algorithm='lbfgs', all_possible_transitions=True)

    print('\nEvaluating CRF')

    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    print(classification_report(y_test, y_pred))

    dict_report = classification_report(y_test, y_pred, output_dict=True)

    data_conll = ''

    for data, real_tags, pred_tags in \
            zip(test_data, y_test, y_pred):
        words = data[0]
        sent = '\n'.join('{0} {1} {2}'.format(word, real_tag, pred_tag)
                         for word, real_tag, pred_tag in
                         zip(words, real_tags, pred_tags))
        sent += '\n\n'
        data_conll += sent

    print(f'\nReport: {dict_report}')

    print(f'\nSaving the report in: {report_file}')

    dump_report(dict_report, report_file)

    script_result_file = os.path.join(report_dir, 'results_crf.tsv')

    with open(script_result_file, 'w', encoding='utf-8') as file:
        file.write(data_conll)
