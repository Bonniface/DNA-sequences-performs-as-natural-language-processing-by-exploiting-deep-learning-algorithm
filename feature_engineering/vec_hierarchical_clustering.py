import pandas as pd
import numpy as np
import ray
import psutil
import os
import argparse
from . import common
from . import hierarchical_clustering


def vec_hierarchical_clustering_add_args(parser):

    parser.add_argument('train_file_path',
                        help='The preprocessed train data file path')
    parser.add_argument('vectors_file_path',
                        help='The embedding npy file')
    parser.add_argument('--cluster_col',
                        help='Column to save cluster id, default is "prot2vec_cluster_id"',
                        default='prot2vec_cluster_id')
    parser.add_argument('--affinity',
                        help='What distance metric to use, default is "euclidean"',
                        default='euclidean')
    parser.add_argument('--linkage',
                        help='What linkage to use single/average/complete, default is "complete"',
                        default='complete')
    parser.add_argument('--cutoff_factor',
                        help='cutoff factor for the hierarchical clustering cutoff distance, default is 0.65.',
                        type=float, default=0.65)
    parser.add_argument('--num_cpus',
                        help='How many cpus are available for ray threads',
                        type=int)
    parser.add_argument('--thread_memory',
                        help='thread memory size for ray.init()',
                        type=int)

    parser.set_defaults(func=vec_hierarchical_clustering_exec)


def vec_hierarchical_clustering_exec(args):

    train_file_path = args.train_file_path
    if not os.path.isfile(train_file_path):
        raise Exception('Data file error, make sure data file path: {}'.format(train_file_path))

    vectors_file_path = args.vectors_file_path
    if not os.path.isfile(vectors_file_path):
        raise Exception('Data file error, make sure data file path: {}'.format(vectors_file_path))

    data_file = pd.read_csv(train_file_path, sep='\t')
    print('Data file loaded')
    required_columns = ["disease_diagnosis", "subject_id", "v_gene", "j_gene", "cdr3_aa", "cdr3_aa_length"]
    for column in required_columns:
        if column not in data_file.columns:
            raise Exception('Data file error, {} is not in data_file columns, use prepare_data_file.'.format(column))

    vectors_file = np.load(vectors_file_path)
    print('Vectors file loaded')
    if vectors_file.shape[0] != data_file.shape[0]:
        raise Exception('mismatch between data_file length ({}) and vectors_file length'.format(
            vectors_file.shape[0] != data_file.shape[0]))

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

    def get_vectors(frame_index) -> np.array:
        return vectors_file[frame_index, :]

    def get_cut_off_dist(frame_vectors: np.array, agg_idx) -> int:
        return common.get_frame_cutoff_factor(frame_vectors, args.affinity, args.cutoff_factor)

    data_file = hierarchical_clustering.find_clusters(data_file, args.cluster_col, args.affinity, args.linkage,
                                                      get_vectors, get_cut_off_dist)
    print('saving clusters inplace')
    data_file.to_csv(train_file_path, sep='\t', index=False)


def main():

    parser = argparse.ArgumentParser('Find clusters using hierarchical clustering on vectors frames grouped by the '
                                     ' V-gene, J-gene and cdr3 length of their origin sequences.')
    vec_hierarchical_clustering_add_args(parser)
    args = parser.parse_args()
    vec_hierarchical_clustering_exec(args)


if __name__ == '__main__':
    main()
