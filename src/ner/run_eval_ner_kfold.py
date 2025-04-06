import os

from seqeval.metrics import classification_report
from src.ner.ner_utils import dump_report


if __name__ == '__main__':

    model_folders_dir = '../../data/corpus/v2/models/ner_simp'
    results_dir = '../../data/corpus/v2/results_simp_kfold/ner'

    list_folder_names = os.listdir(model_folders_dir)

    list_folder_names.sort()

    for folder_name in list_folder_names:

        print(f'\nFolder: {folder_name}')

        results_folder_dir = os.path.join(results_dir, f'{folder_name}')

        os.makedirs(results_folder_dir, exist_ok=True)

        model_dir = os.path.join(model_folders_dir, folder_name)

        list_model_names = os.listdir(model_dir)

        list_model_names.sort()

        for model_name in list_model_names:

            tsv_file_path = os.path.join(model_dir, model_name, 'test.tsv')

            if not os.path.exists(tsv_file_path):
                continue

            print(f'\n\tModel Name: {model_name}')

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

            dict_report = classification_report(all_y_test, all_y_pred, output_dict=True)

            results_file_path = os.path.join(results_folder_dir, f'{model_name}.csv')

            dump_report(dict_report, results_file_path)

            tsv_result_file = os.path.join(results_folder_dir, f'{model_name}.tsv')

            with open(tsv_result_file, 'w', encoding='utf-8') as file:
                file.write(data_tsv)
