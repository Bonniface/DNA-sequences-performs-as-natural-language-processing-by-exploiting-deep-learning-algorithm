import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os


def add_common_arguments(parser):

    parser.add_argument('train_features_file',
                        help='The feature table of the train subjects')
    parser.add_argument('test_features_file',
                        help='The feature table of the test subjects"')
    parser.add_argument('train_labels_file',
                        help='The labels table of the train subjects')
    parser.add_argument('test_labels_file',
                        help='The labels table of the test subjects')
    parser.add_argument('--subject_col',
                        help='Name of the column to index the samples, default: "subject_id"',
                        default="subject_id")
    parser.add_argument('--use_cv',
                        help='Optimize the classifier hyper parameters using cross validation. Default is False.',
                        type=lambda x: x.lower() in ("yes", "true", "t", "1"), default=False)
    parser.add_argument('--labels_to_classify', help='Semicolon separated list of labels to classify, labels not in the'
                                                     'list will be classified "Neutral", default: all labels')


def read_common_args(args):
    train_features_file = args.train_features_file
    test_features_file = args.test_features_file
    train_labels_file = args.train_labels_file
    test_labels_file = args.test_labels_file

    if not os.path.isfile(train_features_file):
        raise Exception('Train features file error, make sure data file path: {}\nExiting...'.format(train_features_file))

    if not os.path.isfile(test_features_file):
        raise Exception('Test features file error, make sure data file path: {}\nExiting...'.format(test_features_file))

    if not os.path.isfile(train_labels_file):
        raise Exception('Train features file error, make sure data file path: {}\nExiting...'.format(train_labels_file))

    if not os.path.isfile(test_labels_file):
        raise Exception('Test features file error, make sure data file path: {}\nExiting...'.format(test_labels_file))

    subject_col = args.subject_col

    train_features = pd.read_csv(train_features_file, index_col=subject_col, sep='\t')
    test_features = pd.read_csv(test_features_file, index_col=subject_col, sep='\t')
    train_labels = pd.read_csv(train_labels_file, index_col=subject_col, sep='\t')
    test_labels = pd.read_csv(test_labels_file, index_col=subject_col, sep='\t')

    labels_to_classify = args.labels_to_classify
    if labels_to_classify is None:
        labels_to_classify = train_labels.iloc[:, 0].unique()
    else:
        labels_to_classify = labels_to_classify.split(';')
        train_labels.iloc[(~train_labels.iloc[:, 0].isin(labels_to_classify)).to_list(), 0] = 'Neutral'

    # anyway we cannot classify a label we didn't see in the
    if not set(test_labels.iloc[:, 0].unique()).issubset(set(labels_to_classify)):
        test_labels.iloc[(~test_labels.iloc[:, 0].isin(labels_to_classify)).to_list(), 0] = 'Neutral'

    if ('Neutral' in train_labels.iloc[:, 0]) or ('Neutral' in test_labels.iloc[:, 0]):
        labels_to_classify += ['Neutral']

    return train_features, test_features, train_labels, test_labels, labels_to_classify


def classify(train_features: pd.DataFrame,
             test_features: pd.DataFrame,
             train_labels: pd.DataFrame,
             test_labels: pd.DataFrame,
             clf,
             labels_to_classify: list) -> (pd.DataFrame(), pd.DataFrame()):

    clf.fit(train_features, train_labels)
    predict = clf.predict(test_features)

    report = classification_report(test_labels, predict, zero_division=0, output_dict=True)
    print(report)
    output = pd.DataFrame()
    for key, item in report.items():
        if key == 'accuracy':
            output.loc[0, key] = item
            continue
        output.loc[0, key + ' precision'] = item['precision']
        output.loc[0, key + ' recall'] = item['recall']
        output.loc[0, key + ' f1-score'] = item['f1-score']
        output.loc[0, key + ' support'] = item['support']
    output = output.sort_index(axis=1)

    confusion_mat = confusion_matrix(test_labels, predict, labels=labels_to_classify)
    confusion_mat = pd.DataFrame(confusion_mat, index=labels_to_classify,
                                 columns=labels_to_classify).sort_index(axis=1).sort_index(axis=0)

    print(output.transpose())
    print(confusion_mat)

    return clf, output, confusion_mat

