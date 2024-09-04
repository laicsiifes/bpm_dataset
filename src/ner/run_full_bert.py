import os
import numpy as np
import matplotlib.pyplot as plt

from src.ner.bert_utils import read_corpus_file, extract_labels, replace_labels, tokenize_and_align_labels
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification,
                          EarlyStoppingCallback)
from transformers import TrainingArguments, Trainer
from seqeval.metrics import classification_report
from src.ner.ner_utils import dump_report
from sklearn.metrics import ConfusionMatrixDisplay


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions_ = np.argmax(logits, axis=-1)
    # Remove ignored index (special tokens) and convert to labels
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

    # model_name = 'distilbert'
    # model_name = 'bert_base'
    # model_name = 'roberta_base'
    # model_name = 'bert_large'
    model_name = 'roberta_large'

    num_epochs = 100

    batch_size = 32

    corpus_dir = '../../data/corpus/conll/'
    model_dir = '../../data/models/ner'
    results_dir = '../../data/results/ner/'

    train_file = '../../data/corpus/conll/train.conll'
    eval_file = '../../data/corpus/conll/validation.conll'
    test_file = '../../data/corpus/conll/test.conll'

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

    model_dir = os.path.join(model_dir, f'full_{model_name}')

    os.makedirs(model_dir, exist_ok=True)

    training_model_dir = os.path.join(model_dir, 'training')
    best_model_dir = os.path.join(model_dir, 'best_model')

    os.makedirs(training_model_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    results_dir = os.path.join(results_dir, f'full_{model_name}')

    os.makedirs(results_dir, exist_ok=True)

    train_data = read_corpus_file(train_file, delimiter='\t', ner_column=1)
    eval_data = read_corpus_file(eval_file, delimiter='\t', ner_column=1)
    test_data = read_corpus_file(test_file, delimiter='\t', ner_column=1)

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

    print(f'\nTrain: {len(train_ds)}')
    print(f'Eval: {len(eval_ds)}')
    print(f'Test: {len(test_ds)}\n')

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
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
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
        tokenizer=tokenizer,
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

    print('\n\n', dict_report)

    report_file = os.path.join(results_dir, f'full_{model_name}.csv')

    dump_report(dict_report, report_file)

    plt.rcParams.update({'font.size': 10})

    list_y_test = []

    for y in true_test_labels:
        list_y_test.extend(y)

    list_y_pred = []

    for y in predicted_test_labels:
        list_y_pred.extend(y)

    disp = ConfusionMatrixDisplay.from_predictions(list_y_test, list_y_pred, xticks_rotation=90)

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

    img_path = os.path.join(results_dir, confusion_matrix_name)

    plt.savefig(img_path, dpi=300)

