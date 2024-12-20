import os
import torch
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

from src.corpus.corpus_utils import read_relations_csv
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import f1_score, classification_report
from src.ner.ner_utils import dump_report
from sklearn.metrics import ConfusionMatrixDisplay


def tokenize_text(examples_, tokenizer_, max_len_):
    return tokenizer_(examples_['text'], padding='max_length', max_length=max_len_, truncation=True)


def compute_metrics_classification(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels = np.argmax(labels, axis=-1)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    return {
        'f1_macro': f1_macro
    }


if __name__ == '__main__':

    folders_dir = '../../data/corpus/v2/folders'
    model_dir = '../../data/corpus/v2/models/relations'
    results_dir = '../../data/corpus/v2/results_kfold/relations'
    overall_results_dir = '../../data/corpus/v2/overall_results/relations'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(overall_results_dir, exist_ok=True)

    num_epochs = 20

    max_len = 512

    batch_size = 16

    learning_rate = 5e-5

    gradient_accumulation_steps = 1
    gradient_checkpointing = False
    fp16 = False
    optim = 'adamw_torch'

    # model_name = 'distilbert'
    # model_name = 'bert_base'
    # model_name = 'roberta_base'
    # model_name = 'bert_large'
    model_name = 'roberta_large'

    if model_name == 'distilbert':
        model_checkpoint = 'distilbert/distilbert-base-cased'
    elif model_name == 'bert_base':
        model_checkpoint = 'google-bert/bert-base-cased'
    elif model_name == 'bert_large':
        model_checkpoint = 'google-bert/bert-large-cased'
        learning_rate = 1e-5
    elif model_name == 'roberta_base':
        model_checkpoint = 'FacebookAI/roberta-base'
    elif model_name == 'roberta_large':
        model_checkpoint = 'FacebookAI/roberta-large'
        learning_rate = 1e-5
    else:
        print('Model Name Option Invalid!')
        exit(0)

    print(f'\nModel Name: {model_name}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'\nDevice: {device} -- {learning_rate}')

    print('\nRunning BERT Relations Extraction Experiment')

    list_folder_names = os.listdir(folders_dir)

    list_folder_names.sort()

    list_y_test = []
    list_y_pred = []

    for folder_name in list_folder_names:

        print(f'\n\tFolder: {folder_name}')

        train_file_path = os.path.join(folders_dir, folder_name, f'train_rel_{folder_name}.csv')
        val_file_path = os.path.join(folders_dir, folder_name, f'val_rel_{folder_name}.csv')
        test_file_path = os.path.join(folders_dir, folder_name, f'test_rel_{folder_name}.csv')

        train_texts, train_labels = read_relations_csv(train_file_path)
        validation_texts, validation_labels = read_relations_csv(val_file_path)
        test_texts, test_labels = read_relations_csv(test_file_path)

        print(f'\n\t\tTrain: {len(train_texts)} -- {len(train_labels)} -- {Counter(train_labels)}')
        print(f'\t\tValidation: {len(validation_texts)} -- {len(validation_labels)} -- {Counter(validation_labels)}')
        print(f'\t\tTest: {len(test_texts)} -- {len(test_labels)} -- {Counter(test_labels)}')

        label_encoder = LabelEncoder()

        y_train = label_encoder.fit_transform(train_labels)
        y_validation = label_encoder.transform(validation_labels)
        y_test = label_encoder.transform(test_labels)

        y_train = torch.tensor(y_train)
        y_validation = torch.tensor(y_validation)
        y_test = torch.tensor(y_test)

        num_classes = len(set(train_labels))

        print(f'\n\t\tLabels Mappings: {num_classes} -- {label_encoder.classes_}')

        y_train = f.one_hot(y_train.to(torch.int64), num_classes=num_classes)
        y_validation = f.one_hot(y_validation.to(torch.int64), num_classes=num_classes)
        y_test = f.one_hot(y_test.to(torch.int64), num_classes=num_classes)

        train_dict = {'text': train_texts, 'label': y_train}
        valid_dict = {'text': validation_texts, 'label': y_validation}
        test_dict = {'text': test_texts, 'label': y_test}

        train_dataset = Dataset.from_dict(train_dict)
        valid_dataset = Dataset.from_dict(valid_dict)
        test_dataset = Dataset.from_dict(test_dict)

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        encoded_train_dataset = train_dataset.map(lambda x: tokenize_text(x, tokenizer, max_len),
                                                  batched=True, batch_size=batch_size)
        encoded_valid_dataset = valid_dataset.map(lambda x: tokenize_text(x, tokenizer, max_len),
                                                  batched=True, batch_size=batch_size)
        encoded_test_dataset = test_dataset.map(lambda x: tokenize_text(x, tokenizer, max_len),
                                                batched=True, batch_size=batch_size)

        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_classes)

        output_dir = os.path.join(model_dir, f'{folder_name}', f'{model_name}', 'training')
        best_model_dir = os.path.join(model_dir, f'{folder_name}', f'{model_name}', 'best_model')

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)

        training_args = TrainingArguments(output_dir=output_dir, logging_strategy='epoch',
                                          gradient_accumulation_steps=gradient_accumulation_steps,
                                          gradient_checkpointing=gradient_checkpointing,
                                          fp16=fp16, optim=optim, weight_decay=0.01, eval_steps=100,
                                          logging_steps=100, learning_rate=learning_rate,
                                          eval_strategy='epoch',
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          num_train_epochs=num_epochs, save_total_limit=2,
                                          save_strategy='epoch', load_best_model_at_end=True,
                                          metric_for_best_model='f1_macro', greater_is_better=True,
                                          report_to=['none'])

        trainer = Trainer(model=model, args=training_args, train_dataset=encoded_train_dataset,
                          eval_dataset=encoded_valid_dataset,
                          compute_metrics=compute_metrics_classification,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

        trainer.train()

        trainer.save_model(best_model_dir)

        y_pred, _, _ = trainer.predict(encoded_test_dataset)

        y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        y_test = [int(y.item()) for y in y_test]
        y_pred = [int(y.item()) for y in y_pred]

        y_test = label_encoder.inverse_transform(y_test)
        y_pred = label_encoder.inverse_transform(y_pred)

        dict_report = classification_report(y_true=y_test, y_pred=y_pred, zero_division=0,
                                            output_dict=True)

        print(f'\n\t\t\tReport: {dict_report}')

        results_folder_dir = os.path.join(results_dir, f'{folder_name}')

        os.makedirs(results_folder_dir, exist_ok=True)

        report_file = os.path.join(results_folder_dir, f'{model_name}_{folder_name}.csv')

        dump_report(dict_report, report_file)

        for y in y_test:
            list_y_test.append(y)

        for y in y_pred:
            list_y_pred.append(y)

    overall_dict_report = classification_report(y_true=list_y_test, y_pred=list_y_pred, zero_division=0,
                                        output_dict=True)

    report_file = os.path.join(overall_results_dir, f'{model_name}_overall.csv')

    dump_report(overall_dict_report, report_file)

    plt.rcParams.update({'font.size': 6})

    disp = ConfusionMatrixDisplay.from_predictions(list_y_test,
                                                   list_y_pred,
                                                   xticks_rotation=90)

    fig = disp.ax_.get_figure()

    fig.set_figwidth(6)

    fig.set_figheight(5)

    plt.subplots_adjust(left=0.25, bottom=0.35)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.ylabel('Labels Reais', fontsize=10)
    plt.xlabel('Labels Preditas', fontsize=10)

    confusion_matrix_name = f'{model_name}_confusion_matrix.pdf'

    confusion_matrix_name = confusion_matrix_name.lower()

    img_path = os.path.join(overall_results_dir, confusion_matrix_name)

    plt.savefig(img_path, dpi=300)


