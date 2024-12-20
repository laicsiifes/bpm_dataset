import pandas as pd


def read_corpus_file(corpus_file, delimiter=' ', ner_column=1, is_unify_tags=False):
    with open(corpus_file, encoding='utf-8') as file:
        lines = file.readlines()
    data = {
        'tokens': [],
        'labels': []
    }
    words = []
    labels = []
    for line in lines:
        line = line.replace('\n', '')
        if line != '':
            if delimiter in line:
                fragments = line.split(delimiter)
                words.append(fragments[0])
                label = fragments[ner_column]
                if is_unify_tags:
                    if 'B-trigger' == label or 'B-catch' == label:
                        label = 'B-activity'
                    elif 'I-trigger' == label or 'I-catch' == label:
                        label = 'I-activity'
                labels.append(label)
        else:
            if len(words) > 1:
                data['tokens'].append(words)
                data['labels'].append(labels)

            words = []
            labels = []
    return data


def extract_labels(labels: list):
    """
    Extrair todas as possíveis labels do dataset
    :param labels: Lista com as labels do dataset
    :return: all_labels = [list]
    """
    unified = []
    for li in labels:
        unified.extend(li)
    df = pd.DataFrame({'all labels': unified})
    all_labels = df['all labels'].value_counts()
    all_labels = list(all_labels.index)
    # Ordenando labels (O, B-, I-, B-, I-,...)
    b_prefix = [label for label in all_labels if label[0] == 'B']
    i_prefix = [label for label in all_labels if label[0] == 'I']
    b_prefix.sort()
    i_prefix.sort()
    all_labels_sort = []
    for i in range(len(b_prefix)):
        all_labels_sort.append(b_prefix[i])
        all_labels_sort.append(i_prefix[i])
    all_labels_sort.insert(0, 'O')
    return all_labels_sort


def replace_labels(dataset, labelpid):
    ds = dataset.copy()
    for li_labels in ds['labels']:
        for i, label in enumerate(li_labels):
            li_labels[i] = labelpid[label]
    return ds


def align_labels_with_tokens(ner_ids, word_ids):
    new_ner_ids = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            ner_id = -100 if word_id is None else ner_ids[word_id]
            new_ner_ids.append(ner_id)
        elif word_id is None:
            new_ner_ids.append(-100)
        else:
            ner_id = ner_ids[word_id]
            if ner_id % 2 == 1:
                ner_id += 1
            new_ner_ids.append(ner_id)
    return new_ner_ids


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, max_length=512, is_split_into_words=True
    )
    all_labels = examples['ner_tags']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs

