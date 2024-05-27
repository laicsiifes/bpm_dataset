import os

from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import SGD


if __name__ == '__main__':

    corpus_dir = '../../data/corpus/conll/'

    report_dir = '../../data/reports/ner/bilstm_crf'
    model_dir = '../../data/models/ner'

    list_embedding_names = [
        # 'glove',
        'flair',
        # 'bert_base'
    ]

    is_use_crf = True

    n_epochs = 100

    batch_size = 32

    columns_dict = {
        0: 'token',
        1: 'label'
    }

    train_file = f'train.conll'
    val_file = f'validation.conll'
    test_file = f'test.conll'

    list_embedding = []

    if 'glove' in list_embedding_names:
        glove_embeddings = WordEmbeddings('glove')
        list_embedding.append(glove_embeddings)

    if 'flair' in list_embedding_names:
        forward_flair_embeddings = FlairEmbeddings('news-forward')
        backward_flair_embeddings = FlairEmbeddings('news-backward')
        list_embedding.append(forward_flair_embeddings)
        list_embedding.append(backward_flair_embeddings)

    if 'bert_base' in list_embedding_names:
        bert_embeddings = TransformerWordEmbeddings('google-bert/bert-base-cased')
        list_embedding.append(bert_embeddings)

    print(f'\nList Embeddings: {list_embedding_names}')

    stacked_embeddings = StackedEmbeddings(list_embedding)

    embedding_model_name = '_'.join(list_embedding_names)

    model_dir = os.path.join(model_dir, embedding_model_name)
    report_dir = os.path.join(report_dir, embedding_model_name)

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    corpus = ColumnCorpus(corpus_dir, columns_dict, train_file=train_file, test_file=test_file,
                          dev_file=val_file)

    print(f'\nTrain len: {len(corpus.train)}')
    print(f'Dev len: {len(corpus.dev)}')
    print(f'Test len: {len(corpus.test)}')

    print(f"\nTrain: {corpus.train[0].to_tagged_string('label')}")
    print(f"Dev: {corpus.dev[0].to_tagged_string('label')}")
    print(f"Test: {corpus.test[0].to_tagged_string('label')}\n")

    tag_type = 'label'

    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    print(f'\nTags: {tag_dictionary.idx2item}')

    tagger = SequenceTagger(hidden_size=256, embeddings=stacked_embeddings, tag_dictionary=tag_dictionary,
                            tag_type=tag_type, use_crf=is_use_crf)

    trainer = ModelTrainer(tagger, corpus)

    trainer.train(base_path=model_dir, optimizer=SGD, learning_rate=0.1, patience=20,
                  mini_batch_size=batch_size, max_epochs=n_epochs)
