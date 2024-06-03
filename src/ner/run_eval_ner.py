import os
import pandas as pd
import matplotlib.pyplot as plt

from seqeval.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from src.ner.ner_utils import dump_report


if __name__ == '__main__':

    model_dir = '../../data/models/ner'

    report_dir = '../../data/reports/ner/bilstm_crf'

    os.makedirs(report_dir, exist_ok=True)

    list_model_names = os.listdir(model_dir)

    columns_names = ['tokens', 'real_tag', 'predicted_tag']

    for model_name in list_model_names:

        print(f'\nModel: {model_name}')

        tsv_file = os.path.join(model_dir, model_name, 'test.tsv')

        if not os.path.exists(tsv_file):
            continue

        with open(file=tsv_file, mode='r') as file:
            lines = file.readlines()

        all_y_test = []
        all_y_pred = []

        y_test = []
        y_pred = []

        for line in lines:

            line = line.replace('\n', '').strip()

            fragments = line.split(' ')

            if len(fragments) == 3:
                y_test.append(fragments[1])
                y_pred.append(fragments[2])
            else:
                all_y_test.append(y_test.copy())
                all_y_pred.append(y_pred.copy())
                y_test.clear()
                y_pred.clear()

        dict_report = classification_report(all_y_test, all_y_pred, output_dict=True)

        results_file_path = os.path.join(report_dir, f'results_{model_name}.csv')

        dump_report(dict_report, results_file_path)

        list_y_test = []

        for y in all_y_test:
            list_y_test.extend(y)

        list_y_pred = []

        for y in all_y_pred:
            list_y_pred.extend(y)

        ConfusionMatrixDisplay.from_predictions(list_y_test, list_y_pred, xticks_rotation='vertical')

        confusion_matrix_name = f'{model_name}_confusion_matrix.pdf'

        confusion_matrix_name = confusion_matrix_name.lower()

        img_path = os.path.join(report_dir, confusion_matrix_name)

        plt.savefig(img_path, dpi=300)
