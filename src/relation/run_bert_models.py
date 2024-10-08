import numpy as np
import torch
import torch.nn.functional as f

from src.corpus.corpus_utils import read_relations_csv
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import f1_score, classification_report



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

    num_epochs = 2

    max_len = 512

    batch_size = 16

    gradient_accumulation_steps = 1
    gradient_checkpointing = False
    fp16 = False
    optim = 'adamw_torch'

    train_file_path = '../../data/corpus/v2/train_rel.csv'
    val_file_path = '../../data/corpus/v2/validation_rel.csv'
    test_file_path = '../../data/corpus/v2/test_rel.csv'

    # model_name = 'distilbert'
    model_name = 'bert_base'
    # model_name = 'roberta_base'
    # model_name = 'bert_large'
    # model_name = 'roberta_large'

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

    train_texts, train_labels = read_relations_csv(train_file_path)
    validation_texts, validation_labels = read_relations_csv(val_file_path)
    test_texts, test_labels = read_relations_csv(test_file_path)

    print(f'\nTrain: {len(train_texts)} -- {len(train_labels)} -- {Counter(train_labels)}')
    print(f'Validation: {len(validation_texts)} -- {len(validation_labels)} -- {Counter(validation_labels)}')
    print(f'Test: {len(test_texts)} -- {len(test_labels)} -- {Counter(test_labels)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'\nDevice: {device}')

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(train_labels)
    y_validation = label_encoder.transform(validation_labels)
    y_test = label_encoder.transform(test_labels)

    y_train = torch.tensor(y_train)
    y_validation = torch.tensor(y_validation)
    y_test = torch.tensor(y_test)

    num_classes = len(set(train_labels))

    print(f'\nLabels Mappings: {num_classes} -- {label_encoder.classes_}')

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

    training_args = TrainingArguments(output_dir=output_dir, logging_strategy='epoch',
                                      gradient_accumulation_steps=gradient_accumulation_steps,
                                      gradient_checkpointing=gradient_checkpointing,
                                      fp16=fp16, optim=optim, weight_decay=0.01, eval_steps=100,
                                      logging_steps=100, learning_rate=5e-5,
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

    print(y_test[:50])
    print('\n\n', y_pred[:50])

    y_test = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(y_pred)

    report = classification_report(y_true=test_labels, y_pred=y_pred, zero_division=0)

    print(report)
