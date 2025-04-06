import os
import optuna

from src.corpus.corpus_utils import read_corpus_file, unify_tags
from src.ner.ner_features import data_preprocessing, convert_data
from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report
from src.ner.ner_utils import dump_report


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

    is_unify_tags = True

    folders_dir = '../../data/corpus/v2/folders'

    if is_unify_tags:
        results_dir = '../../data/corpus/v2/results_simp_kfold/ner'
    else:
        results_dir = '../../data/corpus/v2/results_kfold/ner'

    os.makedirs(results_dir, exist_ok=True)

    delimiter = '\t'

    ner_column = 1

    list_folder_names = os.listdir(folders_dir)

    list_folder_names.sort()

    print('\n\nRunning CRF NER Experiment')

    for folder_name in list_folder_names:

        print(f'\n\tFolder: {folder_name}')

        results_folder_dir = os.path.join(results_dir, f'{folder_name}')

        os.makedirs(results_folder_dir, exist_ok=True)

        train_file = os.path.join(folders_dir, f'{folder_name}', f'train_{folder_name}.conll')
        val_file = os.path.join(folders_dir, f'{folder_name}', f'val_{folder_name}.conll')
        test_file = os.path.join(folders_dir, f'{folder_name}', f'test_{folder_name}.conll')

        train_data = read_corpus_file(train_file, delimiter=delimiter, ner_column=ner_column)
        validation_data = read_corpus_file(val_file, delimiter=delimiter, ner_column=ner_column)
        test_data = read_corpus_file(test_file, delimiter=delimiter, ner_column=ner_column)

        if is_unify_tags:
            train_data = unify_tags(train_data)
            validation_data = unify_tags(validation_data)
            test_data = unify_tags(test_data)

        print(f'\n\t\tTrain data: {len(train_data)}')
        print(f'\t\tValidation data: {len(validation_data)}')
        print(f'\t\tTest data: {len(test_data)}')

        print('\n\t\tPreprocessing ...')

        print('\n\t\t\tTrain data')

        train_data = data_preprocessing(train_data)

        print('\t\t\tValidation data')

        validation_data = data_preprocessing(validation_data)

        print('\t\t\tTest data')

        test_data = data_preprocessing(test_data)

        X_train, y_train = convert_data(train_data)
        X_val, y_val = convert_data(validation_data)
        X_test, y_test = convert_data(test_data)

        print(f'\n\t\tExample features: {X_train[-1]}')
        print(f'\t\tTags: {y_train[-1]}')

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

        print(f'\n\t\tTrain Labels: {len(all_train_labels)} -- {all_train_labels}')
        print(f'\t\tValidation Labels: {len(all_validation_labels)} -- {all_validation_labels}')
        print(f'\t\tTest Labels: {len(all_test_labels)} -- {all_test_labels}')

        study = optuna.create_study(direction='maximize')

        study.optimize(objective, n_trials=10)

        print('\n\t\tBest trial: ')

        best_trial = study.best_trial

        print(f'\n\t\t\tValue: {best_trial.value:.3f}')

        print('\t\t\tParams: ')

        for key, value in best_trial.params.items():
            print(f'\t\t\t\t{key}: {value}')

        crf = CRF(**best_trial.params, algorithm='lbfgs', all_possible_transitions=True)

        print('\n\t\tEvaluating CRF')

        crf.fit(X_train, y_train)

        y_pred = crf.predict(X_test)

        dict_report = classification_report(y_test, y_pred, output_dict=True)

        data_tsv = ''

        for data, real_tags, pred_tags in zip(test_data, y_test, y_pred):
            words = data[0]
            sent = '\n'.join('{0} {1} {2}'.format(word, real_tag, pred_tag)
                             for word, real_tag, pred_tag in
                             zip(words, real_tags, pred_tags))
            sent += '\n\n'
            data_tsv += sent

        print(f'\n\t\t\tReport: {dict_report}')

        report_file = os.path.join(results_folder_dir, f'crf_{folder_name}.csv')

        print(f'\n\t\tSaving the report in: {report_file}')

        dump_report(dict_report, report_file)

        tsv_result_file = os.path.join(results_folder_dir, f'crf_{folder_name}.tsv')

        with open(tsv_result_file, 'w', encoding='utf-8') as file:
            file.write(data_tsv)
