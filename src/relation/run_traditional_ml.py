from src.corpus.corpus_utils import read_relations_csv
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


if __name__ == '__main__':

    train_file_path = '../../data/corpus/v2/train_rel.csv'
    val_file_path = '../../data/corpus/v2/validation_rel.csv'
    test_file_path = '../../data/corpus/v2/test_rel.csv'

    max_features = None

    train_texts, train_labels = read_relations_csv(train_file_path)
    validation_texts, validation_labels = read_relations_csv(val_file_path)
    test_texts, test_labels = read_relations_csv(test_file_path)

    print(f'\nTrain: {len(train_texts)} -- {len(train_labels)} -- {Counter(train_labels)}')
    print(f'Validation: {len(validation_texts)} -- {len(validation_labels)} -- {Counter(validation_labels)}')
    print(f'Test: {len(test_texts)} -- {len(test_labels)} -- {Counter(test_labels)}')

    vectorizer = TfidfVectorizer(analyzer='word', binary=False, ngram_range=(3, 3),
                                 max_features=max_features)

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(train_labels)
    y_val = label_encoder.transform(validation_labels)
    y_test = label_encoder.transform(test_labels)

    print(f'\nLabels Mappings: {label_encoder.classes_}')

    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_val = vectorizer.transform(validation_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    classifiers = {
        'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=500),
        'knn': KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(class_weight='balanced'),
        'random_forest': RandomForestClassifier(class_weight='balanced'),
        'extra_trees_classifier': ExtraTreesClassifier(class_weight='balanced'),
        'svc': SVC(class_weight='balanced'),
        'svc_ovr': SVC(class_weight='balanced', decision_function_shape='ovr'),
        'mlp_classifier': MLPClassifier()
    }

    print('\n\n------------Evaluations------------\n')

    for clf_name, classifier in classifiers.items():

        print(f'\n\n  Classifier: {clf_name}')

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        print('\t', y_pred[:20])
        print('\t', y_test[:20])

        clf_report = classification_report(y_test, y_pred)

        print(clf_report)




