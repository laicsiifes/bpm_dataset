def save_conll_file(list_sentences: list, conll_file_path: str):
    conll_file = ''
    for sentence in list_sentences:
        tokens = [s[2] for s in sentence[2]]
        labels = [s[3] for s in sentence[2]]
        # print(f'\nTokens: {tokens}')
        # print(f'Labels: {labels}')
        for token, label in zip(tokens, labels):
            if '\n' not in token:
                conll_file += f'{token}\t{label}\n'
        conll_file += '\n'
    with open(file=conll_file_path, mode='w', encoding='utf-8') as file:
        file.write(conll_file)


def read_corpus_file(corpus_file: str, delimiter: str = '\t', ner_column: int = 1) -> list:
    with open(corpus_file, encoding='utf-8') as file:
        lines = file.readlines()
    data = []
    words = []
    tags = []
    for line in lines:
        line = line.replace('\n', '')
        if line != '':
            if delimiter in line:
                fragments = line.split(delimiter)
                words.append(fragments[0])
                tags.append(fragments[ner_column])
        else:
            if len(words) > 1:
                data.append((words, tags))
            words = []
            tags = []
    return data
