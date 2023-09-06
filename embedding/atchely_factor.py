# This is an implementation of the embedding methods described in https://doi.org/10.1093/bioinformatics/btu523

from itertools import product
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


rf_random_grid = {'n_estimators': [100, 200, 300],
                  'max_features': [2, 4, 'auto'],
                  'max_depth': [10, 20, 30, 'auto']}

dt_random_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': [2, 4, 6, 8, 10]}

knn_grid = {"n_neighbors": [1, 3, 5, 8]}

# create mapping from AA kmer (k=3) to feature for the based on kmeans of all possible 3-mers of AA embedded by their
# atchley factors
atchley_props = pd.DataFrame(list(map(lambda x: x.split(","), "A,-0.591,-1.302,-0.733,1.570,-0.146;C,-1.343,0.465,-0.862,-1.020,-0.255;D,1.050,0.302,-3.656,-0.259,-3.242;E,1.357,-1.453,1.477,0.113,-0.837;F,-1.006,-0.590,1.891,-0.397,0.412;G,-0.384,1.652,1.330,1.045,2.064;H,0.336,-0.417,-1.673,-1.474,-0.078;I,-1.239,-0.547,2.131,0.393,0.816;K,1.831,-0.561,0.533,-0.277,1.648;L,-1.019,-0.987,-1.505,1.266,-0.912;M,-0.663,-1.524,2.219,-1.005,1.212;N,0.945,0.828,1.299,-0.169,0.933;P,0.189,2.081,-1.628,0.421,-1.392;Q,0.931,-0.179,-3.005,-0.503,-1.853;R,1.538,-0.055,1.502,0.440,2.897;S,-0.228,1.399,-4.760,0.670,-2.647;T,-0.032,0.326,2.213,0.908,1.313;V,-1.337,-0.279,-0.544,1.242,-1.262;W,-0.595,0.009,0.672,-2.128,-0.184;Y,0.260,0.830,3.097,-0.838,1.512".split(";"))),
                        columns=["amino.acid", "f1", "f2", "f3", "f4", "f5"]).set_index("amino.acid")
all_aa_3kmers = list(map(lambda x: ''.join(x), product('ACDEFGHIKLMNPQRSTVWY', repeat=3)))
kmer2Index = pd.Series(range(len(all_aa_3kmers)), index=all_aa_3kmers)
atchley_kmer_embedding = []
for kmer in kmer2Index.index:
    l = []
    for c in kmer:
        l.extend(atchley_props.loc[c, :].to_list())
    atchley_kmer_embedding.append(l)
atchley_kmer_embedding = np.array(atchley_kmer_embedding)
atchley_kmeans = KMeans(n_clusters=100)
atchley_kmeans.fit(atchley_kmer_embedding)
kmer2feature = pd.Series(atchley_kmeans.predict(atchley_kmer_embedding), index=all_aa_3kmers)


def embed_sequences_atchely_factors(df):

    df = df[df.cdr3_aa_trimmed.str.find('X') < 0].reset_index()
    kmers = [[kmer2feature["%s%s%s" % (c1,c2,c3)] for c1,c2,c3 in
              zip(seq, seq[1:], seq[2:])] for seq in df.cdr3_aa_trimmed]

    vectors = np.zeros(shape=(len(df), 100))
    for i, features in enumerate(kmers):
        features_cnt = pd.value_counts(features)
        vectors[i, features_cnt.index.values] = features_cnt.values

    return df, vectors