import numpy as np
import os
import pandas as pd
from . import common
import argparse
from sklearn.metrics.pairwise import pairwise_distances


def build_feature_list_add_args(parser):

    parser.add_argument('train_file_path',
                        help='The preprocessed train data file path')
    parser.add_argument('vectors_file_path',
                        help='The embedding npy file')
    parser.add_argument('output_desc',
                        help='Description prefix for the output file names')
    parser.add_argument('output_folder_path',
                        help='Output folder to save output files')
    parser.add_argument('--metric',
                        help='What distance metric to use, default is "euclidean"',
                        default='euclidean')
    parser.add_argument('--cluster_col',
                        help='Column to save cluster id, default is "prot2vec_cluster_id"',
                        default='prot2vec_cluster_id')
    parser.add_argument('--target_labels',
                        help='Semicolon separated list of labels to be used for the significance calculation the '
                             'features, default: all labels')
    parser.add_argument('--cutoff_factor',
                        help='cutoff factor for the hierarchical clustering cutoff distance, default is 0.65.',
                        type=float, default=0.65)
    parser.add_argument('--cluster_size_th',
                        help='th filter for minimal number of sequences in cluster, default is 20',
                        default=10, type=int)
    parser.add_argument('--subjects_th',
                        help='th filter by min subjects contributing sequences to teh cluster, default is 2',
                        default=2, type=int)
    parser.add_argument('--significance_th', help='th filter - how significant is the dominant label sum of '
                                                  'frequencies squares, compare to the other labels. Default is 1.5.',
                        default=1.5, type=float)

    parser.set_defaults(func=build_vec_feature_list_exec)


def build_vec_feature_list_exec(args):

    train_file_path = args.train_file_path
    if not os.path.isfile(train_file_path):
        raise Exception('Data file error, make sure data file path: {}'.format(train_file_path))
    print('data file loaded')
    data_file = pd.read_csv(train_file_path, sep='\t')

    required_columns = ["disease_diagnosis", "subject_id", "v_gene", "j_gene", "cdr3_aa", "cdr3_aa_length"]
    for column in required_columns:
        if column not in data_file.columns:
            raise Exception('Data file error, {} is not in data_file columns, use prepare_data_file.'.format(column))

    vectors_file_path = args.vectors_file_path
    if not os.path.isfile(vectors_file_path):
        raise Exception('Data file error, make sure data file path: {}'.format(vectors_file_path))
    vectors_file = np.load(vectors_file_path)
    print('Vectors file loaded')

    if vectors_file.shape[0] != data_file.shape[0]:
        raise Exception('mismatch between data_file length ({}) and vectors_file length'.format(
            vectors_file.shape[0] != data_file.shape[0]))

    output_desc = args.output_desc
    output_folder_path = args.output_folder_path
    cutoff_factor = args.cutoff_factor
    subjects_th = args.subjects_th
    significance_th = args.significance_th
    target_labels = args.target_labels
    cluster_col = args.cluster_col

    if target_labels is not None:
        target_labels = target_labels.split(';')
    else:
        # the target labels are the labels for which we would like to find high significance clusters
        target_labels = data_file['disease_diagnosis'].unique()

    def get_max_dist(frame_index):
        cdr3_aa_length = data_file.loc[frame_index, 'cdr3_aa_length'].iloc[0]
        v_gene = data_file.loc[frame_index, 'v_gene'].iloc[0]
        j_gene = data_file.loc[frame_index, 'j_gene'].iloc[0]
        vectors_frame = vectors_file[
                        (data_file.cdr3_aa_length == cdr3_aa_length) & (data_file.v_gene == v_gene) & (
                                data_file.j_gene == j_gene), :]
        return common.get_frame_cutoff_factor(vectors_frame, args.metric, cutoff_factor)

    selected_clusters = common.build_feature_list(
        data_file, get_max_dist, cluster_col, 'subject_id', 'disease_diagnosis', subjects_th, significance_th,
        args.cluster_size_th, target_labels)

    output_file = os.path.join(output_folder_path, output_desc + '_features.tsv')
    selected_clusters.to_csv(output_file, sep='\t', index=False)
    print('feature list is saved to {}'.format(output_file))


def main():

    parser = argparse.ArgumentParser('Clusters are filtered to features based on 3 threshold filters: subjects_th, '
                                     'significance_th and cluster_size_th.')
    build_feature_list_add_args(parser)
    args = parser.parse_args()
    build_vec_feature_list_exec(args)


if __name__ == '__main__':
    main()
