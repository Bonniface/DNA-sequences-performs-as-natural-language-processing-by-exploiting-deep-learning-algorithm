import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import pairwise_distances


def get_frame_cutoff_factor(vectors_frame, metric, cutoff_factor):
    norm_vec = np.std(vectors_frame, 0)
    zero_vec = np.zeros(vectors_frame.shape[1])
    dist_std = pairwise_distances(norm_vec.reshape(1, -1), zero_vec.reshape(1, -1), metric=metric)[0, 0]
    return dist_std * cutoff_factor


def build_feature_list(data_file: pd.DataFrame, get_max_dist, cluster_id_column, subject_col, label_col,
                       subjects_th, significance_th, cluster_size_th, target_labels: pd.Series) -> pd.DataFrame:
    selected_clusters = pd.DataFrame(columns=['cluster_id', 'max_distance', 'v_gene', 'j_gene',
                                              'cdr3_aa_length', 'num_subjects', 'label', 'significance'])

    # filter clusters which their size is below the minimum filter
    clusters_freq = data_file[cluster_id_column].value_counts()
    clusters_freq = clusters_freq[clusters_freq > cluster_size_th]
    data_file = data_file[data_file[cluster_id_column].isin(clusters_freq.keys().tolist())]

    t0 = time.time()
    for cluster_id, frame in data_file.groupby([cluster_id_column]):

        frame_subjects = frame.loc[:, ].groupby(by=[subject_col])[label_col].apply(lambda x: x.iloc[0])
        num_subjects = len(frame_subjects)
        # filter clusters with not enough sample representation
        if num_subjects < subjects_th:
            continue

        frame_subject_fraction = frame[subject_col].value_counts(normalize=True)
        # if more than 90% of the sequences come from the same subject - skip
        if frame_subject_fraction.values[0] >= 0.9:
            continue

        frame_subject_freq = frame[subject_col].value_counts()
        # square the frequencies to reward high frequencies
        frame_subject_freq = frame_subject_freq ** 2

        frame_labels = frame_subjects.unique()
        # no representation for the target label/labels in this cluster
        if len(set(frame_labels) & set(target_labels)) == 0:
            continue

        # check if the target labeling representation is distinguished enough
        if len(frame_labels) > 1:
            labels_freq = pd.Series(
                list(map(lambda x: frame_subject_freq[frame_subjects.index[frame_subjects == x]].sum(),
                         frame_labels)), index=frame_labels).sort_values(ascending=False)
            # filter this cluster if the dominated label is not a target label
            if labels_freq.index[0] not in target_labels:
                continue
            # filter this cluster if the dominated label is not significant enough compare to the next label
            if labels_freq.iloc[0] < labels_freq.iloc[1] * significance_th:
                continue
            significance = labels_freq.iloc[0]
            label = labels_freq.index[0]

        else:
            significance = frame_subject_freq.sum()
            label = frame_labels[0]

        max_distance = get_max_dist(frame.index)
        vgene = frame['v_gene'].iloc[0]
        jgene = frame['j_gene'].iloc[0]
        cdr3_aa_length = frame['cdr3_aa_length'].iloc[0]
        selected_clusters.loc[len(selected_clusters), :] = [cluster_id, max_distance, vgene, jgene,
                                                            cdr3_aa_length, num_subjects, label, significance]

    print('{} out of {} clusters passed the filtering after {} msec'.format(len(selected_clusters),
                                                                            len(data_file[cluster_id_column].unique()),
                                                                            time.time() - t0))
    return selected_clusters

