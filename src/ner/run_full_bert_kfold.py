import os
import numpy as np

from src.ner.bert_utils import (read_corpus_file, extract_labels, replace_labels,
                                tokenize_and_align_labels)
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, EarlyStoppingCallback)
from transformers import TrainingArguments, Trainer
from seqeval.metrics import classification_report
from src.ner.ner_utils import dump_report


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions_ = np.argmax(logits, axis=-1)
    true_labels = [[label_names[k] for k in label if k != -100] for label in labels]
    predicted_labels = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_, labels)
    ]
    all_metrics_ = classification_report(true_labels, predicted_labels, output_dict=True)
    metrics = {
        'precision': all_metrics_['micro avg']['precision'],
        'recall': all_metrics_['micro avg']['recall'],
        'f1': all_metrics_['micro avg']['f1-score']
    }
    return metrics


if __name__ == '__main__':

    is_unify_tags = False

    folders_dir = '../../data/corpus/v2/folders'

    if is_unify_tags:
        model_dir = '../../data/corpus/v2/models/ner_simp'
        results_dir = '../../data/corpus/v2/results_simp_kfold/ner'
    else:
        model_dir = '../../data/corpus/v2/models/ner'
        results_dir = '../../data/corpus/v2/results_kfold/ner'

    # model_name = 'distilbert'
    # model_name = 'bert_base'
    # model_name = 'roberta_base'
    # model_name = 'bert_large'
    model_name = 'roberta_large'

    num_epochs = 100

    batch_size = 32

    learning_rate = 2e-5

    os.makedirs(results_dir, exist_ok=True)

    model_checkpoint = None

    if model_name == 'distilbert':
        model_checkpoint = 'distilbert/distilbert-base-cased'
    elif model_name == 'bert_base':
        model_checkpoint = 'google-bert/bert-base-cased'
    elif model_name == 'bert_large':
        model_checkpoint = 'google-bert/bert-large-cased'
    elif model_name == 'roberta_base':
        model_checkpoint = 'FacebookAI/roberta-base'
    elif model_name == 'roberta_large':
        model_checkpoint = 'FacebookAI/roberta-large'
    else:
        print('Model Name Option Invalid!')
        exit(0)

    print(f'\nModel: {model_name}')

    list_folder_names = os.listdir(folders_dir)

    list_folder_names.sort()

    print('\n\nRunning Full BERT NER Experiment')

    for folder_name in list_folder_names:

        print(f'\n\tFolder: {folder_name}')

        train_file = os.path.join(f'{folders_dir}', f'{folder_name}', f'train_{folder_name}.conll')
        eval_file = os.path.join(f'{folders_dir}', f'{folder_name}', f'val_{folder_name}.conll')
        test_file = os.path.join(f'{folders_dir}', f'{folder_name}', f'test_{folder_name}.conll')

        model_folder_dir = os.path.join(model_dir, f'{folder_name}', f'full_{model_name}')

        os.makedirs(model_folder_dir, exist_ok=True)

        training_model_dir = os.path.join(model_folder_dir, 'training')
        best_model_dir = os.path.join(model_folder_dir, 'best_model')

        os.makedirs(training_model_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)

        results_folder_dir = os.path.join(results_dir, f'{folder_name}')

        os.makedirs(results_folder_dir, exist_ok=True)

        train_data = read_corpus_file(train_file, delimiter='\t', ner_column=1, is_unify_tags=is_unify_tags)
        eval_data = read_corpus_file(eval_file, delimiter='\t', ner_column=1, is_unify_tags=is_unify_tags)
        test_data = read_corpus_file(test_file, delimiter='\t', ner_column=1, is_unify_tags=is_unify_tags)

        label_names = extract_labels(labels=train_data['labels'])

        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {label: i for i, label in id2label.items()}

        train_data = replace_labels(train_data, label2id)
        test_data = replace_labels(test_data, label2id)
        eval_data = replace_labels(eval_data, label2id)

        train_ds = Dataset.from_dict({
            'tokens': train_data['tokens'],
            'ner_tags': train_data['labels']
        })

        test_ds = Dataset.from_dict({
            'tokens': test_data['tokens'],
            'ner_tags': test_data['labels']
        })

        eval_ds = Dataset.from_dict({
            'tokens': eval_data['tokens'],
            'ner_tags': eval_data['labels']
        })

        raw_datasets = DatasetDict({
            'train': train_ds,
            'test': test_ds,
            'validation': eval_ds
        })

        print(f'\n\t\tTrain: {len(train_ds)}')
        print(f'\t\tEval: {len(eval_ds)}')
        print(f'\t\tTest: {len(test_ds)}\n')

        if 'roberta_' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            batch_size=batch_size,
            remove_columns=raw_datasets['train'].column_names,
            fn_kwargs={
                'tokenizer': tokenizer
            }
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer
        )

        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id
        )

        logging_eval_steps = 100

        args = TrainingArguments(
            output_dir=training_model_dir,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            eval_steps=logging_eval_steps,
            logging_steps=logging_eval_steps,
            save_total_limit=1,
            weight_decay=0.01,
            push_to_hub=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
        )

        trainer.train()

        trainer.save_model(best_model_dir)

        predictions = trainer.predict(test_dataset=tokenized_datasets['test'])

        predictions = np.argmax(predictions.predictions, axis=-1)

        test_labels = tokenized_datasets['test']['labels']

        true_test_labels = [[label_names[k] for k in label if k != -100] for label in test_labels]

        predicted_test_labels = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, test_labels)
        ]

        dict_report = classification_report(true_test_labels, predicted_test_labels, output_dict=True)

        print(f'\n\t\t\tReport: {dict_report}')

        report_file = os.path.join(results_folder_dir, f'full_{model_name}_{folder_name}.csv')

        dump_report(dict_report, report_file)

        data_tsv = ''

        for real_tags, pred_tags in zip(true_test_labels, predicted_test_labels):
            sent = '\n'.join('TOKEN {0} {1}'.format(real_tag, pred_tag)
                             for real_tag, pred_tag in
                             zip(real_tags, pred_tags))
            sent += '\n\n'
            data_tsv += sent

        script_result_file = os.path.join(results_folder_dir, f'full_{model_name}_{folder_name}.tsv')

        with open(script_result_file, 'w', encoding='utf-8') as file:
            file.write(data_tsv)
