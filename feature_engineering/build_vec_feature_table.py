import pandas as pd
import numpy as np
import ray
from sklearn.metrics.pairwise import pairwise_distances
import os
import psutil
import argparse


def build_feature_table_add_args(parser):

    parser.add_argument('train_file_path',
                        help='The preprocessed train data file path')
    parser.add_argument('test_file_path',
                        help='The preprocessed test data file path')
    parser.add_argument('train_vectors_file_path',
                        help='The embedding train npy file')
    parser.add_argument('test_vectors_file_path',
                        help='The embedding test npy file')
    parser.add_argument('feature_file_path',
                        help='The clusters (features) file')
    parser.add_argument('output_desc',
                        help='Description prefix for the output file names')
    parser.add_argument('output_folder_path',
                        help='Output folder to save output files')
    parser.add_argument('--metric',
                        help='What distance metric to use, default is "euclidean"',
                        default='euclidean')
    parser.add_argument('--num_cpus',
                        help='How many cpus are available for ray threads',
                        type=int)
    parser.add_argument('--thread_memory',
                        help='thread memory size for ray.init()',
                        type=int)
    parser.add_argument('--subject_col',
                        help='name of the column with the subject identifier, default: "subject_id"',
                        default="subject_id")

    parser.set_defaults(func=build_vec_feature_table_exec)


def build_vec_feature_table_exec(args):

    train_file_path = args.train_file_path
    test_file_path = args.test_file_path
    train_vectors_file_path = args.train_vectors_file_path
    test_vectors_file_path = args.test_vectors_file_path
    feature_file_path = args.feature_file_path
    output_desc = args.output_desc
    output_folder_path = args.output_folder_path
    subject_col = args.subject_col

    if not os.path.isfile(train_file_path):
        raise Exception('Data file error, make sure data file path: {}'.format(train_file_path))
    if not os.path.isfile(test_file_path):
        raise Exception('Data file error, make sure data file path: {}'.format(test_file_path))

    if not os.path.isfile(train_vectors_file_path):
        raise Exception('Data file error, make sure vectors file path: {}'.format(train_vectors_file_path))
    if not os.path.isfile(test_vectors_file_path):
        raise Exception('Data file error, make sure vectors file path: {}'.format(test_vectors_file_path))

    if not os.path.isfile(feature_file_path):
        raise Exception('Data file error, make sure data file path: {}'.format(feature_file_path))

    num_cpus = args.num_cpus
    if num_cpus is None:
        num_cpus = psutil.cpu_count()

    thread_memory = args.thread_memory

    try:
        if thread_memory is not None and thread_memory > 0:
            ray.init(num_cpus=num_cpus, lru_evict=True, memory=thread_memory, object_store_memory=thread_memory,
                     ignore_reinit_error=True)
        else:
            ray.init(num_cpus=num_cpus, lru_evict=True, ignore_reinit_error=True)
    except:
        if thread_memory is not None and thread_memory > 0:
            ray.init(num_cpus=num_cpus, memory=thread_memory, object_store_memory=thread_memory,
                     ignore_reinit_error=True)
        else:
            ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    train_data = pd.read_csv(train_file_path, sep='\t')
    print('Data file {} loaded'.format(train_file_path))
    test_data = pd.read_csv(test_file_path, sep='\t')
    print('Data file {} loaded'.format(test_file_path))
    train_vectors = np.load(train_vectors_file_path)
    print('Vectors file {} loaded'.format(train_vectors_file_path))
    test_vectors = np.load(test_vectors_file_path)
    print('Vectors file {} loaded'.format(test_vectors_file_path))

    if train_vectors.shape[0] != train_data.shape[0]:
        raise Exception('mismatch between train data_file length ({}) and vectors_file length'.format(
            train_vectors.shape[0] != train_data.shape[0]))

    if test_vectors.shape[0] != test_data.shape[0]:
        raise Exception('mismatch between test data_file length ({}) and vectors_file length'.format(
            test_vectors.shape[0] != test_data.shape[0]))

    feature_file = pd.read_csv(feature_file_path, sep='\t')
    print('Features file {} loaded'.format(feature_file_path))

    train_features, train_labels, test_features, test_labels = build_feature_table(train_data, test_data, train_vectors,
                                           test_vectors, feature_file, args.metric, subject_col)

    train_features_output_file = os.path.join(output_folder_path, output_desc + '_train_features.tsv')
    train_labels_output_file = os.path.join(output_folder_path, output_desc + '_train_labels.tsv')
    train_features.to_csv(train_features_output_file, sep='\t', index=False)
    train_labels.to_csv(train_labels_output_file, sep='\t', index=False)

    test_features_output_file = os.path.join(output_folder_path, output_desc + '_test_features.tsv')
    test_labels_output_file = os.path.join(output_folder_path, output_desc + '_test_labels.tsv')
    test_features.to_csv(test_features_output_file, sep='\t', index=False)
    test_labels.to_csv(test_labels_output_file, sep='\t', index=False)

    print('train and test features and labels files are saved to: {}'.format(output_folder_path))


def build_feature_table(train_data: pd.DataFrame, test_data: pd.DataFrame, train_vectors: np.array,
                        test_vectors: np.array, features_file: pd.DataFrame, metric, subject_col='subject_id',
                        label_col='disease_diagnosis') -> pd.DataFrame:

    train_data_id = ray.put(train_data)
    test_data_id = ray.put(test_data)
    train_vectors_id = ray.put(train_vectors)
    test_vectors_id = ray.put(test_vectors)
    features_file_id = ray.put(features_file)

    train_features_table = pd.DataFrame(
        0, index=train_data[subject_col].unique(), columns=features_file['cluster_id'].values)

    for subject in train_data[subject_col].unique():
        subject_data = train_data[train_data[subject_col] == subject]
        train_features_table.loc[subject, :] = list(map(
            lambda x: sum(subject_data['immune2vec_cluster_id'] == x) ** 2, train_features_table.columns))
        train_features_table.loc[subject, label_col] = subject_data[label_col].iloc[0]
        train_features_table.loc[subject, subject_col] = subject

    result_ids = []
    for subject in test_data[subject_col].unique():
        result_ids += [get_subject_feature_table.remote(subject, train_data_id, test_data_id, train_vectors_id,
                                                        test_vectors_id, features_file_id, metric,
                                                        subject_col, label_col)]

    test_features_table = pd.concat([ray.get(res_id) for res_id in result_ids])

    return (train_features_table.drop(columns=[label_col]),
            train_features_table[[subject_col, label_col]],
            test_features_table.drop(columns=[label_col]),
            test_features_table[[subject_col, label_col]])


@ray.remote
def get_subject_feature_table(subject: str, train_file: pd.DataFrame, test_file: pd.DataFrame, train_vectors: np.array,
                              test_vectors: np.array, features_file: pd.DataFrame, metric, subject_col: str,
                              label_col: str) -> pd.DataFrame:

    print('Start creating feature table for subject {}'.format(subject))

    # filter data for specific subject
    subject_data = test_file[test_file[subject_col] == subject]

    subject_features_table = pd.DataFrame(0, index=[subject], columns=features_file['cluster_id'].values)
    sum_features_count = 0
    num_features_count = 0

    # group the features by their gene assignment and cdr3 length
    for agg_idx, feature_frame in features_file.groupby(['v_gene', 'j_gene', 'cdr3_aa_length']):

        subject_frame = subject_data[(subject_data['v_gene'] == agg_idx[0]) &
                                     (subject_data['j_gene'] == agg_idx[1]) &
                                     (subject_data['cdr3_aa_length'] == agg_idx[2])]
        if len(subject_frame) == 0:
            continue

        # all the sequences of the feature clusters with the specific v_gene, j_gene and cdr3_aa_length
        indexing = [False] * len(train_file)
        for cluster_id in feature_frame.cluster_id.unique():
            indexing |= train_file['immune2vec_cluster_id'] == cluster_id
        train_frame = train_file.loc[indexing, :]

        subject_vectors = test_vectors[subject_frame.index, :]
        feature_vectors = train_vectors[train_frame.index, :]

        # compute the distance matrix
        distances = pairwise_distances(X=subject_vectors, Y=feature_vectors, metric=metric)

        for cluster_id in feature_frame.cluster_id.unique():
            # filter the columns relevant to sequences of the specific cluster id
            cluster_distances = distances[:, train_frame['immune2vec_cluster_id'] == cluster_id]
            # flag all distances which are within the clustering cutoff value
            distance_close_enough = cluster_distances <= feature_frame.loc[feature_frame.cluster_id == cluster_id,
                                                                           'max_distance'].iloc[0]
            # we are counting sequences which are within the cutoff distance from all the cluster members
            # meaning they are members of the cluster (by complete linkage rule)
            cluster_freq = np.sum(np.equal(distance_close_enough.sum(axis=1), cluster_distances.shape[1]))
            if cluster_freq == 0:
                continue
            # square the result to reward high frequencies
            subject_features_table.loc[subject, cluster_id] = cluster_freq ** 2
            num_features_count += 1
            sum_features_count += cluster_freq ** 2

    print('Finished creating feature table for subject {} label {}: Number of none zero features {}, Sum none '
          'zero features: {}'.format(subject, subject_data[label_col].iloc[0], num_features_count, sum_features_count))

    subject_features_table[label_col] = subject_data[label_col].iloc[0]
    subject_features_table[subject_col] = subject

    return subject_features_table


def main():

    parser = argparse.ArgumentParser(description='Build repertoire feature table based on the repertoire frequency in '
                                                 'the clusters.')
    build_feature_table_add_args(parser)
    args = parser.parse_args()
    build_vec_feature_table_exec(args)


if __name__ == '__main__':
    main()
