import os
import pandas as pd
import pathlib
import numpy as np
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold


def split_dataset_folds_add_args(parser):

    parser.add_argument('data_file_path',
                        help='The data file to split')
    parser.add_argument('output_desc',
                        help='Description prefix for the test/train files name')
    parser.add_argument('output_folder_path',
                        help='Output directory where the folds directories will be created')
    parser.add_argument('--n_splits',
                        help='Number of CV splits (without reputation), default=10',
                        type=int, default=10)
    parser.add_argument('--n_repeats',
                        help='Number of CV repeats, default=10',
                        type=int, default=10)
    parser.add_argument('--start_fold',
                        help='Number of the fold to start from - used for extend existing split with more folds, '
                             'default=0',
                        type=int, default=0)
    parser.add_argument('--test_size',
                        help='The proportion of the data set, default=0.1 - note that this value should align with the '
                             'n_splits argument, meaning n_splits <= (1/test_size). Default is 0.1. Only relevant if'
                             ' the shuffle_splits argument is True, else test size is controlled by the number of '
                             'splits.',
                        type=float, default=0.1)
    parser.add_argument('--shuffle_labels',
                        help='Create the splits with shuffled labels - for benchmarking against random labeling, '
                             'default is False',
                        type=lambda x: x.lower() in ("yes", "true", "t", "1"), default=False)
    parser.add_argument('--shuffle_splits', help='Randomize the splits - a chance of the same split repeating itself, '
                                                 ' but with ability to control test_size',
                        type=lambda x: x.lower() in ("yes", "true", "t", "1"), default=False)
    parser.add_argument('--stratified', help='split the train/test folds with the same labels proportion as in the '
                                             'dataset. Default is True.',
                        type=lambda x: x.lower() in ("yes", "true", "t", "1"), default=True)

    parser.set_defaults(func=split_dataset_folds_exec)


def split_dataset_folds_exec(args):

    data_file_path = args.data_file_path

    if not os.path.isfile(data_file_path):
        raise Exception('Data file error, make sure data file path: {}'.format(data_file_path))

    data_file = pd.read_csv(data_file_path, sep='\t')
    print('Data file loaded')

    if 'disease_diagnosis' not in data_file.columns:
        raise Exception('"disease_diagnosis" is not in {} columns'.format(data_file_path))

    id_column = 'subject_id'
    if id_column not in data_file.columns:
        raise Exception('"subject_id" is not in {} columns'.format(data_file_path))

    split_folds(data_file, args.output_folder_path, args.output_desc,
                args.n_splits, args.n_repeats, args.test_size, args.shuffle_labels,
                args.stratified, args.shuffle_splits)


def split_folds(data_file: pd.DataFrame,
                output_folder_path: str, output_desc: str,
                n_splits: int, n_repeats: int, test_size: float,
                shuffle_labels: bool, stratified: bool, shuffle_splits: bool,
                subject_col='subject_id', label_col='disease_diagnosis'):

    subjects = data_file.loc[:, ].groupby(by=[subject_col])[label_col].apply(lambda x: x.iloc[0])
    print('total number of subject is {}'.format(len(subjects)))
    # sort index to ensure getting the same splits in different runs
    subjects.sort_index(inplace=True)

    if shuffle_labels is True:
        print("shuffling labels before split")
        np.random.seed(0)
        subjects[:] = np.random.permutation(subjects)
        for subject_id, label in subjects.iteritems():
            data_file.loc[data_file[subject_col] == subject_id, label_col] = label

    split_id = 0

    if shuffle_splits is False:
        if stratified is True:
            kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        else:
            kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

        for train, test in kf.split(subjects.index, subjects.values):
            save_fold(train, test, data_file, subjects, subject_col, output_folder_path, output_desc, split_id)
            split_id += 1

        return

    for random_state in range(start_fold, start_fold + n_repeats):

        if stratified is True:
            kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        else:
            kf = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        for train, test in kf.split(subjects.index, subjects.values):
            save_fold(train, test, data_file, subjects, subject_col, output_folder_path, output_desc, split_id)
            split_id += 1


def save_fold(train, test, data_file, subjects, subject_col, output_folder_path, output_desc, split_id):

    train_subjects = subjects.iloc[train]
    train_output = data_file.loc[data_file[subject_col].isin(train_subjects.index), :]

    test_subjects = subjects.iloc[test]
    test_output = data_file.loc[data_file[subject_col].isin(test_subjects.index), :]

    output_dir = os.path.join(output_folder_path, 'split' + str(split_id))
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_output.to_csv(os.path.join(output_dir, output_desc + '_train.tsv'), sep='\t', index=False)
    test_output.to_csv(os.path.join(output_dir, output_desc + '_test.tsv'), sep='\t', index=False)


def main():

    parser = argparse.ArgumentParser(description='Split a data file to train and test data sets')
    split_dataset_folds_add_args(parser)
    args = parser.parse_args()
    split_dataset_folds_exec(args)


if __name__ == '__main__':
    main()

