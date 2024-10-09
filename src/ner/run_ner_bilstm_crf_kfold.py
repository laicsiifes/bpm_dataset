import os

from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import SGD


if __name__ == '__main__':

    folders_dir = '../../data/corpus/v2/folders'
    model_dir = '../../data/corpus/v2/models/ner'

    n_epochs = 100

    is_use_crf = True

    list_folder_names = os.listdir(folders_dir)

    list_folder_names.sort()

    list_embedding_names = [
        'glove',
        # 'flair',
        # 'distilbert',
        # 'bert_base'
    ]

    batch_size = 32

    columns_dict = {
        0: 'token',
        1: 'label'
    }

    list_embedding = []

    if 'glove' in list_embedding_names:
        glove_embeddings = WordEmbeddings('glove')
        list_embedding.append(glove_embeddings)

    if 'flair' in list_embedding_names:
        forward_flair_embeddings = FlairEmbeddings('news-forward')
        backward_flair_embeddings = FlairEmbeddings('news-backward')
        list_embedding.append(forward_flair_embeddings)
        list_embedding.append(backward_flair_embeddings)

    if 'distilbert' in list_embedding_names:
        bert_embeddings = TransformerWordEmbeddings('distilbert/distilbert-base-cased')
        list_embedding.append(bert_embeddings)

    if 'bert_base' in list_embedding_names:
        bert_embeddings = TransformerWordEmbeddings('google-bert/bert-base-cased')
        list_embedding.append(bert_embeddings)

    if 'bert_large' in list_embedding_names:
        bert_embeddings = TransformerWordEmbeddings('google-bert/bert-large-cased')
        list_embedding.append(bert_embeddings)

    if 'roberta_base' in list_embedding_names:
        bert_embeddings = TransformerWordEmbeddings('FacebookAI/roberta-base')
        list_embedding.append(bert_embeddings)

    if 'roberta_large' in list_embedding_names:
        bert_embeddings = TransformerWordEmbeddings('FacebookAI/roberta-large')
        list_embedding.append(bert_embeddings)

    print(f'\nList Embeddings: {list_embedding_names}\n')

    print('\n\nRunning BiLSTM-CRF NER Experiment')

    for folder_name in list_folder_names:

        print(f'\n\tFolder: {folder_name}')

        stacked_embeddings = StackedEmbeddings(list_embedding)

        embedding_model_name = '_'.join(list_embedding_names)

        model_folder_dir = os.path.join(model_dir, f'{folder_name}', f'bilstm_crf_{embedding_model_name}')

        os.makedirs(model_folder_dir, exist_ok=True)

        train_file = f'train_{folder_name}.conll'
        val_file = f'val_{folder_name}.conll'
        test_file = f'test_{folder_name}.conll'

        corpus_folder_dir = os.path.join(folders_dir, f'{folder_name}')

        corpus = ColumnCorpus(corpus_folder_dir, columns_dict, train_file=train_file, test_file=test_file,
                              dev_file=val_file)

        print(f'\n\t\tTrain len: {len(corpus.train)}')
        print(f'\t\tDev len: {len(corpus.dev)}')
        print(f'\t\tTest len: {len(corpus.test)}')

        print(f"\n\t\tTrain: {corpus.train[0].to_tagged_string('label')}")
        print(f"\t\tDev: {corpus.dev[0].to_tagged_string('label')}")
        print(f"\t\tTest: {corpus.test[0].to_tagged_string('label')}\n")

        tag_type = 'label'

        tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

        print(f'\n\t\tTags: {tag_dictionary.idx2item}')

        tagger = SequenceTagger(hidden_size=256, embeddings=stacked_embeddings, tag_dictionary=tag_dictionary,
                                tag_type=tag_type, use_crf=is_use_crf)

        trainer = ModelTrainer(tagger, corpus)

        trainer.train(base_path=model_folder_dir, optimizer=SGD, learning_rate=0.1, patience=10,
                      mini_batch_size=batch_size, max_epochs=n_epochs)