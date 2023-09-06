import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import os
import argparse


def rf_fe_add_args(parser):

    parser.add_argument('train_features_file',
                        help='The cluster frequency table of the train subjects')
    parser.add_argument('test_features_file',
                        help='The cluster frequency table of the test subjects')
    parser.add_argument('train_labels_file',
                        help='The labels table of the train subjects')
    parser.add_argument('output_folder_path',
                        help='Output folder to save output files')

    parser.add_argument('--num_features',
                        help='The number of features to select, the default is the number of samples in the train',
                        type=int)
    parser.add_argument('--subject_col',
                        help='Name of the column to index the samples, default: "subject_id"',
                        default="subject_id")

    parser.set_defaults(func=rf_fe_exec)


def rf_fe_exec(args):

    train_features_file = args.train_features_file
    test_features_file = args.test_features_file
    train_labels_file = args.train_labels_file
    output_folder_path = args.output_folder_path
    subject_col = args.subject_col
    num_features = args.num_features

    if not os.path.isfile(train_features_file):
        raise Exception('Train features file error, make sure data file path: {}\nExiting...'.format(train_features_file))

    if not os.path.isfile(test_features_file):
        raise Exception('Test features error, make sure data file path: {}\nExiting...'.format(test_features_file))

    if not os.path.isfile(train_labels_file):
        raise Exception('Train features file error, make sure data file path: {}\nExiting...'.format(train_labels_file))

    train_feature_table = pd.read_csv(train_features_file, index_col=subject_col, sep='\t')
    test_feature_table = pd.read_csv(test_features_file, index_col=subject_col, sep='\t')
    train_labels = pd.read_csv(train_labels_file, index_col=subject_col, sep='\t')

    selected_features = select_features_random_forest(train_feature_table, train_labels, num_features)
    train_feature_table = train_feature_table.loc[:, selected_features]
    test_feature_table = test_feature_table.loc[:, selected_features]

    # save output
    train_output_file = os.path.join(output_folder_path,
                                     os.path.basename(train_features_file).split('.tsv')[0] + '_rf_fe.tsv')
    train_feature_table.to_csv(train_output_file, sep='\t')

    test_output_file = os.path.join(output_folder_path,
                                    os.path.basename(test_features_file).split('.tsv')[0] + '_rf_fe.tsv')
    test_feature_table.to_csv(test_output_file, sep='\t')

    print('output files were saved to: {} and {}'.format(train_output_file, test_output_file))


def select_features_random_forest(train_features: pd.DataFrame, train_labels: pd.DataFrame,
                                  num_features=None) -> pd.DataFrame:
    if num_features is None:
        num_features = len(train_features)

    data_x = train_features
    data_y = train_labels
    kf = StratifiedShuffleSplit(n_splits=20, test_size=0.1, random_state=0)
    all_importance = pd.DataFrame()
    for train_index, _ in kf.split(data_x, data_y.iloc[:, 0].ravel()):
        rfc = RandomForestClassifier(n_estimators=len(train_features)*2, class_weight='balanced')
        rfc.fit(data_x.iloc[train_index, :], data_y.iloc[train_index, 0])
        importance = pd.DataFrame(
            {'FEATURE': data_x.columns, 'IMPORTANCE': np.round(rfc.feature_importances_, 3)})
        all_importance = all_importance.append(importance)

    all_importance = all_importance.groupby(['FEATURE']).aggregate({'IMPORTANCE': 'mean'}).sort_values(
        by=['IMPORTANCE'], ascending=False)
    selected_features = all_importance.index[0:num_features].to_list()

    print('using random forest cross validation filtered {} features out of {} features'.format(
        len(selected_features), train_features.shape[1]))

    return selected_features


def main():

    parser = argparse.ArgumentParser('Feature elimination using random forest repeated cross validation folds')
    rf_fe_add_args(parser)
    args = parser.parse_args()
    rf_fe_exec(args)


if __name__ == '__main__':
    main()

