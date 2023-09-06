import pandas as pd
import numpy as np
import time
import ray
import psutil
import gc
from sklearn.cluster import AgglomerativeClustering


def find_clusters(data_file: pd.DataFrame, cluster_id_column, affinity, linkage,
                  get_vectors, get_cut_off_dist) -> pd.DataFrame:

    data_file[cluster_id_column] = None

    results_ids = []
    cluster_max_id = 0

    sequences_completed = 0

    t0 = time.time()

    cutoff_thresholds = pd.DataFrame(columns=['v_gene', 'j_gene', 'cdr3_aa_length', 'cutoff_th'])

    for agg_idx, frame in data_file.groupby(['v_gene', 'j_gene', 'cdr3_aa_length']):

        if len(frame) == 1:
            data_file.loc[frame.index, cluster_id_column] = cluster_max_id
            cluster_max_id += 1
            sequences_completed += 1
            continue

        frame_vectors = get_vectors(frame.index)
        frame_distance_th = get_cut_off_dist(frame_vectors, agg_idx)
        cutoff_thresholds.loc[len(cutoff_thresholds), :] = [agg_idx[0], agg_idx[1], agg_idx[2], frame_distance_th]
        results_ids += [(frame.index, do_agglomerative_clustering.remote(frame_vectors, frame_distance_th,
                                                                         affinity, linkage))]
    cutoff_thresholds.to_csv('cutoff_thresholds.tsv', sep='\t', index=False)
    count = 1
    for (frame_index, res_id) in results_ids:

        auto_garbage_collect()

        cluster_labels = np.copy(ray.get(res_id))
        cluster_labels += cluster_max_id
        cluster_max_id += len(cluster_labels)
        data_file.loc[frame_index, cluster_id_column] = cluster_labels

        sequences_completed += len(frame_index)
        if (sequences_completed * 100 / len(data_file)) >= (count * 10):
            count += 1
            print('{:.2f}% of sequences completed after {} msec'.format(sequences_completed * 100 / len(data_file),
                                                                        time.time() - t0))

    print('100% of sequences completed after {} msec'.format(sequences_completed * 100 / len(data_file),
                                                             time.time() - t0))
    print('found {} clusters'.format(sum(data_file[cluster_id_column].value_counts() > 1)))

    return data_file


def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


@ray.remote
def do_agglomerative_clustering(vectors: np.array, distance_threshold, affinity, linkage):

    clustering = AgglomerativeClustering(affinity=affinity,
                                         linkage=linkage,
                                         distance_threshold=distance_threshold,
                                         n_clusters=None).fit(vectors)
    return clustering.labels_







