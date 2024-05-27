import os
import matplotlib.pyplot as plt

from src.corpus.corpus_utils import read_corpus_file
from src.ner.ner_features import data_preprocessing, convert_data
from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report
from src.ner.ner_utils import dump_report


if __name__ == '__main__':

    delimiter = '\t'

    ner_column = 1

    report_dir = '../../data/reports/ner/crf'

    train_file = '../../data/corpus/conll/train.conll'
    test_file = '../../data/corpus/conll/test.conll'

    os.makedirs(report_dir, exist_ok=True)

    report_file = os.path.join(report_dir, 'results_crf.csv')

    train_data = read_corpus_file(train_file, delimiter=delimiter, ner_column=ner_column)
    test_data = read_corpus_file(test_file, delimiter=delimiter, ner_column=ner_column)

    print(f'\n  Train data: {len(train_data)}')
    print(f'  Test data: {len(test_data)}')

    print('\nPreprocessing ...')

    print('\n  Train data')

    train_data = data_preprocessing(train_data)

    print('  Test data')

    test_data = data_preprocessing(test_data)

    X_train, y_train = convert_data(train_data)
    X_test, y_test = convert_data(test_data)

    print(f'\nExample features: {X_train[-1]}')
    print(f'Tags: {y_train[-1]}')

    all_train_labels = []
    for y in y_train:
        all_train_labels.extend(y)
    all_train_labels = set(all_train_labels)

    all_test_labels = []
    for y in y_test:
        all_test_labels.extend(y)
    all_test_labels = set(all_test_labels)

    print(f'\nTrain Labels: {len(all_train_labels)} -- {all_train_labels}')
    print(f'Test Labels: {len(all_test_labels)} -- {all_test_labels}')

    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=1000, all_possible_transitions=True)

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

    from sklearn.metrics import ConfusionMatrixDisplay

    all_y_test = []

    for y in y_test:
        all_y_test.extend(y)

    all_y_pred = []

    for y in y_pred:
        all_y_pred.extend(y)

    ConfusionMatrixDisplay.from_predictions(all_y_test, all_y_pred, xticks_rotation='vertical')

    confusion_matrix_name = f'confusion_matrix.pdf'

    confusion_matrix_name = confusion_matrix_name.lower()

    img_path = os.path.join(report_dir, confusion_matrix_name)

    plt.savefig(img_path, dpi=300)
