import os
import re
import matplotlib.pyplot as plt

from seqeval.metrics import classification_report
from src.ner.ner_utils import dump_report
from sklearn.metrics import ConfusionMatrixDisplay


if __name__ == '__main__':

    results_dir = '../../data/corpus/v2/results_kfold'
    overall_results_dir = '../../data/corpus/v2/overall_results'

    os.makedirs(overall_results_dir, exist_ok=True)

    list_folder_names = os.listdir(results_dir)

    list_folder_names.sort()

    dict_predictions = {}

    for folder_name in list_folder_names:

        print(f'\nFolder: {folder_name}')

        results_folder_dir = os.path.join(results_dir, f'{folder_name}')

        file_names = os.listdir(results_folder_dir)

        list_file_names = [name for name in file_names if name.endswith('.tsv') ]

        for file_name in list_file_names:

            tsv_file_path = os.path.join(results_folder_dir, file_name)

            file_name = file_name.replace('.tsv', '')

            file_name = re.sub(r'_\d+', '', file_name)

            with open(file=tsv_file_path, mode='r') as file:
                data_tsv = file.read()

            lines = data_tsv.split('\n')

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

            if file_name not in dict_predictions:
                dict_predictions[file_name] = {
                    'y_true': [],
                    'y_pred': []
                }

            dict_predictions[file_name]['y_true'].extend(all_y_test)
            dict_predictions[file_name]['y_pred'].extend(all_y_pred)

    for model_name, model_data in dict_predictions.items():

        print(f'\nModel: {model_name} -- {len(model_data["y_true"])} -- {len(model_data["y_pred"])}')

        dict_report = classification_report(model_data['y_true'], model_data['y_pred'], output_dict=True)

        results_file_path = os.path.join(overall_results_dir, f'{model_name}.csv')

        dump_report(dict_report, results_file_path)

        plt.rcParams.update({'font.size': 8})

        list_y_test = []

        for y in model_data['y_true']:
            list_y_test.extend(y)

        list_y_pred = []

        for y in model_data['y_pred']:
            list_y_pred.extend(y)

        disp = ConfusionMatrixDisplay.from_predictions(list_y_test,
                                                       list_y_pred,
                                                       xticks_rotation=90)

        fig = disp.ax_.get_figure()

        fig.set_figwidth(6)

        fig.set_figheight(5)

        plt.subplots_adjust(left=0.25, bottom=0.25)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.ylabel('Labels Reais', fontsize=15)
        plt.xlabel('Labels Preditas', fontsize=15)

        confusion_matrix_name = f'{model_name}_confusion_matrix.pdf'

        confusion_matrix_name = confusion_matrix_name.lower()

        img_path = os.path.join(overall_results_dir, confusion_matrix_name)

        plt.savefig(img_path, dpi=300)
