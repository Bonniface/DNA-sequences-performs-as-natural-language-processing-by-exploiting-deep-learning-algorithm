import sys
import os
import pandas as pd
import numpy as np
import argparse


def split_vectors_folds_add_args(parser):

    parser.add_argument('data_file_path',
                        help='The *_filtered.tsv data file')
    parser.add_argument('vectors_file_path',
                        help='A vector file to split according to the data files split')
    parser.add_argument('folds_dir',
                        help='A root dir containing the data folds, each fold is in a directory split{fold number}')
    parser.add_argument('data_file_desc',
                        help='Description prefix of test/train files name')
    parser.add_argument('--start_fold',
                        help='Number of the fold to start from - used for extend existing split with more vectors '
                             'folds, default=0',
                        type=int, default=0)

    parser.set_defaults(func=split_vectors_folds_exec)


def split_vectors_folds_exec(args):

    vectors_file_path = args.vectors_file_path
    folds_dir = args.folds_dir
    data_file_path = args.data_file_path
    data_file_desc = args.data_file_desc

    if not os.path.isdir(folds_dir):
        print('folds dir error! Make sure the directory exists.\nExiting...')
        sys.exit(1)

    if not os.path.isfile(data_file_path):
        print('data file error! Make sure the file exists and it is *.tsv file.\nExiting...')
        sys.exit(1)
    data_file = pd.read_csv(data_file_path, usecols=['sequence_id', 'subject_id'], sep='\t')
    print('Data file loaded, original file length: ', data_file.shape[0])

    if not os.path.isfile(vectors_file_path):
        print('vectors file {} error! Make sure the file exists and it is *.npy file.\nExiting...'.format(vectors_file_path))
        sys.exit(1)
    vectors_file = np.load(vectors_file_path)
    print('Vectors file loaded, original file length: ', vectors_file.shape[0])

    if vectors_file.shape[0] != data_file.shape[0]:
        print('mismatch between data_file and vectors_file length\nExiting...')
        sys.exit(1)

    all_sequence_ids = data_file.sequence_id.apply(str) + data_file.subject_id.apply(str)
    all_sequence_ids.name = 'id'
    fold = 0

    while True:

        dir_name = os.path.join(folds_dir, 'split' + str(fold))
        if not os.path.isdir(dir_name):
            break

        for split in ['train', 'test']:
            file_name = os.path.join(folds_dir, 'split' + str(fold), data_file_desc + '_' + split + '.tsv')
            split_data = pd.read_csv(file_name, sep='\t', usecols=['sequence_id', 'subject_id'])
            split_sequence_ids = split_data.sequence_id.apply(str) + split_data.subject_id.apply(str)
            split_sequence_ids.name = 'id'

            indexing = pd.merge(all_sequence_ids, split_sequence_ids, how='inner', right_index=True, on='id').index
            vectors = vectors_file[indexing, :]
            vectors_file_name = os.path.join(
                folds_dir, 'split' + str(fold), data_file_desc + '_' + split + '_vectors.npy')
            print('saving: ', vectors_file_name)
            np.save(vectors_file_name, vectors)

        fold += 1


def main():

    parser = argparse.ArgumentParser(description='Split the vector file according to the data train/test folds')
    split_vectors_folds_add_args(parser)
    args = parser.parse_args()
    split_vectors_folds_exec(args)


if __name__ == '__main__':
    main()
