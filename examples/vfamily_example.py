import os
import time
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
from Bio.Seq import Seq
from changeo.Gene import *
from embedding import generate_model
from embedding import sequence_modeling
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from enum import Enum, auto
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from embedding.atchely_factor import embed_sequences_atchely_factors


class DataSet(Enum):
    hcv = auto()
    celiac = auto()
    flu = auto()
    all = auto()


class Classifier(Enum):
    rf = auto()
    dt = auto()
    knn = auto()
    all = auto()

pca_n_comp = 10
test_size = 0.25

def create_trimmed_dataset(dataset_file):
    # read dataset
    df = pd.read_csv(dataset_file, sep='\t')
    # extracted vfamily from v_call
    df['v_family'] = df.v_call.apply(lambda x: getFamily(x, action='set'))

    if 'cdr3_aa' not in df.columns:
        if 'cdr3' not in df.columns:
            if 'junction_aa' not in df.columns:
                if 'junction' not in df.columns:
                    raise Exception('one of the columsn "cdr3", "cdr3_aa", "junction", "junction_aa" is mandatory')
                else:
                    df = df.loc[df.junction.str.len() % 3 == 0, :]
                    df['junction_aa'] = df.junction.apply(lambda x: Seq(x).translate().__str__())
            df['cdr3_aa'] = df.junction_aa.str[1:-1]
        else:
            df = df.loc[df.cdr3.str.len() % 3 == 0, :]
            df.loc[:, 'cdr3_aa'] = df.cdr3.apply(lambda x: Seq(x).translate().__str__())

    # discard none productive aa sequences
    df = df.loc[df.productive.apply(str).str.lower().isin(['t', 'true', '1', 'yes']), :]
    df = df.loc[df.cdr3_aa.str.find('.') < 0, :]
    df = df.loc[df.cdr3_aa.str.find('-') < 0, :]
    df = df.loc[df.cdr3_aa.str.find('*') < 0, :]
    # trim 2, 3 the protein sequence - we need to be left with AA sequence of length at least 3
    # we want to be left after the trim with AA seq of at least 3 AA
    df = df.loc[df.cdr3_aa.str.len() >= 8, :]
    df['cdr3_aa_trimmed'] = df.cdr3_aa.str[2:-3]

    # discard sequences with multiple vfamily assignment
    df = df[df.v_family.apply(lambda x: len(x) == 1)]
    df.v_family = df.v_family.apply(lambda x: x[0])

    df.to_csv(os.path.basename(dataset_file).split('.tsv')[0] + '_prepared.tsv', sep='\t', index=False)
    return df


def create_model(dataset_file, n_dim, load_models):

    model_desc = os.path.basename(dataset_file).split('.tsv')[0] + '_' + str(n_dim) + 'dim_trimmed_cdr3'
    if load_models and os.path.isfile(model_desc + '.model'):
        print('loading model {}'.format(model_desc + '.model'))
        return sequence_modeling.load_protvec(model_desc + '.model')

    # construct an immune2vec embedding model
    class GenerateModelArgs:
        def __init__(self):
            self.data_file_path = dataset_file
            self.output_folder_path = '../vfamily_example.py'
            self.desc = model_desc
            self.aa_col = 'cdr3_aa_trimmed'
            self.n_dim = n_dim
            self.n_gram = 3
            self.corpus_file = None
            self.reading_frame = None
            self.data_fraction = 1.0
            self.seed = 0
    generate_model.generate_model_exec(GenerateModelArgs())

    # load generated model from file
    model = sequence_modeling.load_protvec(model_desc + '.model')
    return model


def embed_sequences_immune2vec(df, model):
    # embed trimmed cdr3 to vectors using the immune2vec
    print('Generating vectors...')
    def embed_data(word):
        try:
            return list(model.to_vecs(word, n_read_frames=3))
        except:
            return np.nan
    vectors = df.cdr3_aa_trimmed.apply(embed_data)
    print('{:.3}% of data not transformed'.format((100 * sum(vectors.isna()) / len(df))))
    # drop the un translated rows from the file
    df = df.drop(vectors[vectors.isna()].index, axis=0)
    vectors = np.array(vectors[vectors.notna()].tolist())
    df.reset_index(inplace=True)
    return df, vectors


def sample_dataset(df, vectors_after_pca, random_state, sample_size):
    # reduce to the 3 most frequent v_families (to allow subsampling of similar size) and subsample an even number from
    # each family
    sampled_df = pd.concat([df[df.v_family == 'IGHV3'].sample(n=sample_size, random_state=random_state),
                            df[df.v_family == 'IGHV4'].sample(n=sample_size, random_state=random_state),
                            df[df.v_family == 'IGHV1'].sample(n=sample_size, random_state=random_state)])
    sampled_df.sort_index(inplace=True)
    sampled_vectors = vectors_after_pca[sampled_df.index, :]
    return sampled_df, sampled_vectors


def reduce_dimensions_pca(vectors, pca_n_comp):
    # reduce vectors dimensionality to pca_n_comp
    # standardize the vectors
    s_vectors = StandardScaler().fit_transform(vectors)
    # Apply the PCA
    pca = PCA(n_components=pca_n_comp)
    pca.fit(s_vectors)
    vectors_after_pca = pca.transform(s_vectors)
    print('Explained variance: {}, summing to {} of the data'.format(pca.explained_variance_ratio_,
                                                                     sum(pca.explained_variance_ratio_)))
    return vectors


def split_train_data(sampled_vectors, random_seed, labels):
    np.random.seed(random_seed)
    test_rows = np.random.choice(range(sampled_vectors.shape[0]), round(test_size * sampled_vectors.shape[0]))
    train_rows = np.arange(sampled_vectors.shape[0])[-test_rows]
    train_vectors = sampled_vectors[train_rows, :]
    test_vectors = sampled_vectors[test_rows, :]
    train_labels = labels[train_rows]
    test_labels = labels[test_rows]

    return train_vectors, test_vectors, train_labels, test_labels


def plot_vectors(vectors, labels):
    vectors_2dim = reduce_dimensions_pca(vectors, 2)
    plt_df = pd.DataFrame(columns=['x', 'y', 'hue'])
    plt_df['x'] = vectors_2dim[:, 0]
    plt_df['y'] = vectors_2dim[:, 1]
    plt_df['hue'] = labels.values
    sns.scatterplot(data=plt_df, x='x', y='y', hue='hue', style='hue', marker=',', linewidth=0, s=4, alpha=0.6)


def test_classifiers(train_vectors, test_vectors, train_labels, test_labels, random_state, corpus, dataset,
                     sample_size, embedding_type, results, args):
    print('dataset: {}, random_state: {}, sample_size: {}, embedding_type: {}, corpus: {}'.format(
        dataset, random_state, sample_size, embedding_type, corpus))

    if args.classifier in [Classifier.all, Classifier.rf]:
        print("Random Forest:")
        if args.optimize:
            rf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=rf_random_grid, n_iter=100,
                                    cv=3, verbose=3, random_state=42, n_jobs=-1)
        else:
            rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid={}, n_jobs=-1)
        rf.fit(train_vectors, train_labels)
        print(rf.best_params_)
        rf_predict = rf.predict(test_vectors)
        print(classification_report(rf_predict, test_labels, labels=[1,3,4], target_names=['IGHV1', 'IGHV3', 'IGHV4']))
        report = classification_report(rf_predict, test_labels, output_dict=True)
        results.loc[len(results), :] = [random_state, dataset, embedding_type, corpus, 'rf',
                                        report['macro avg']['f1-score'], sample_size, rf.best_params_]
        results.to_csv('results_tmp_pid' + str(os.getpid()) + '.csv', index=False)

    if args.classifier in [Classifier.all, Classifier.dt]:
        print("Decision Tree:")
        if args.optimize:
            dt = RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions=dt_random_grid, n_iter=100,
                                    cv=3, verbose=3, random_state=42, n_jobs=-1)
        else:
            dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid={}, n_jobs=-1)
        dt.fit(train_vectors, train_labels)
        print(dt.best_params_)
        dt_predict = dt.predict(test_vectors)
        print(classification_report(dt_predict, test_labels, labels=[1,3,4], target_names=['IGHV1', 'IGHV3', 'IGHV4']))
        report = classification_report(dt_predict, test_labels, output_dict=True)
        results.loc[len(results), :] = [random_state, dataset, embedding_type, corpus, 'dt',
                                        report['macro avg']['f1-score'], sample_size, dt.best_params_]
        results.to_csv('results_tmp_pid' + str(os.getpid()) + '.csv', index=False)

    if args.classifier in [Classifier.all, Classifier.knn]:
        print("KNearest Neighbors:")
        if args.optimize:
            knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_grid, cv=3, verbose=3, n_jobs=-1)
        else:
            knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': [3]}, n_jobs=-1)
        knn.fit(train_vectors, train_labels)
        print(knn.best_params_)
        knn_predict = knn.predict(test_vectors)
        print(classification_report(knn_predict, test_labels, labels=[1,3,4], target_names=['IGHV1', 'IGHV3', 'IGHV4']))
        report = classification_report(knn_predict, test_labels, output_dict=True)
        results.loc[len(results), :] = [random_state, dataset, embedding_type, corpus, 'knn',
                                        report['macro avg']['f1-score'], sample_size, knn.best_params_]
        results.to_csv('results_tmp_pid' + str(os.getpid()) + '.csv', index=False)
    
    return results


def test_immune2vec_100dim(df, model, corpus, dataset, sample_size, results, args):

    desc = 'immune2vec_100dim_corpus_' + corpus + '_ds_' + dataset
    if args.load_vectors and os.path.isfile(desc + '_vectors.npy') and os.path.isfile(desc + '_dataset.tsv'):
        print('loading vectors {}'.format(desc + '_vectors.npy'))
        vectors = np.load(desc + '_vectors.npy')
        embedded_df = pd.read_csv(desc + '_dataset.tsv', sep='\t')
    else:
        embedded_df, vectors = embed_sequences_immune2vec(df, model)
        np.save(desc + '_vectors.npy', vectors)
        embedded_df.to_csv(desc + '_dataset.tsv', sep='\t', index = False)
    vectors_after_pca = reduce_dimensions_pca(vectors, pca_n_comp)
    for random_state in args.random_seeds:
        sampled_df, sampled_vectors = sample_dataset(embedded_df, vectors_after_pca, random_state, sample_size)
        labels = sampled_df.v_family.str[-1].astype(int).to_numpy()
        train_vectors, test_vectors, train_labels, test_labels = split_train_data(sampled_vectors, random_state, labels)
        results = test_classifiers(train_vectors, test_vectors, train_labels, test_labels, random_state, corpus,
                                   dataset, sample_size, 'immune2vec_100dim', results, args)
    return results


def test_immune2vec_10dim(df, model, corpus, dataset, sample_size, results, args):
    desc = 'immune2vec_10dim_corpus_' + corpus + '_ds_' + dataset
    if args.load_vectors and os.path.isfile(desc + '_vectors.npy') and os.path.isfile(desc + '_dataset.tsv'):
        print('loading vectors {}'.format(desc + '_vectors.npy'))
        vectors = np.load(desc + '_vectors.npy')
        embedded_df = pd.read_csv(desc + '_dataset.tsv', sep='\t')
    else:
        embedded_df, vectors = embed_sequences_immune2vec(df, model)
        np.save(desc + '_vectors.npy', vectors)
        embedded_df.to_csv(desc + '_dataset.tsv', sep='\t', index = False)
    for random_state in args.random_seeds:
        sampled_df, sampled_vectors = sample_dataset(embedded_df, vectors, random_state, sample_size)
        labels = sampled_df.v_family.str[-1].astype(int).to_numpy()
        train_vectors, test_vectors, train_labels, test_labels = split_train_data(sampled_vectors, random_state, labels)
        results = test_classifiers(train_vectors, test_vectors, train_labels, test_labels, random_state, corpus,
                                   dataset, sample_size, 'immune2vec_10dim', results, args)
    return results


def test_atchely_factors(df, dataset, sample_size, results, args):
    desc = 'atchely_ds_' + dataset
    if args.load_vectors and os.path.isfile(desc + '_vectors.npy') and os.path.isfile(desc + '_dataset.tsv'):
        print('loading vectors {}'.format(desc + '_vectors.npy'))
        vectors = np.load(desc + '_vectors.npy')
        embedded_df = pd.read_csv(desc + '_dataset.tsv', sep='\t')
    else:
        embedded_df, vectors = embed_sequences_atchely_factors(df)
        np.save(desc + '_vectors.npy', vectors)
        embedded_df.to_csv(desc + '_dataset.tsv', sep='\t', index = False)
    vectors_after_pca = reduce_dimensions_pca(vectors, pca_n_comp)
    for random_state in args.random_seeds:
        sampled_df, sampled_vectors = sample_dataset(embedded_df, vectors_after_pca, random_state, sample_size)
        labels = sampled_df.v_family.str[-1].astype(int).to_numpy()
        train_vectors, test_vectors, train_labels, test_labels = split_train_data(sampled_vectors, random_state, labels)
        results = test_classifiers(train_vectors, test_vectors, train_labels, test_labels, random_state, None, dataset,
                                   sample_size, 'atchely_factors', results, args)
    return results


def main():

    parser = argparse.ArgumentParser('Run the immune2vec vfamily classifier model in number of configurations.')
    parser.add_argument('--load_models',
                        help='Trying loading model form disk before generating it, default is "True".',
                        type=lambda x: x.lower() in (['1', 't', 'true']), default=True)
    parser.add_argument('--load_vectors',
                        help='Trying loading vectors file from disk before generating it, default is "True".',
                        type=lambda x: x.lower() in (['1', 't', 'true']), default=True)
    parser.add_argument('--optimize',
                        help='Try optimizing the classifier hyper parameters using CV on the train set, default is "False".',
                        type=lambda x: x.lower() in (['1', 't', 'true']), default=False)
    parser.add_argument('--dataset',
                        help='On which dataset to run the model with (hcv/celiac/flu/all), default is "all".',
                        type=lambda x: DataSet.__dict__.get(x), default=DataSet.all)
    parser.add_argument('--classifier',
                        help='Which classifier model (dt/rf/knn) to use, default is "all".',
                        type=lambda x: Classifier.__dict__.get(x), default=Classifier.all)
    parser.add_argument('--random_seeds',
                        help='a space separated list of random seeds, default is "42".',
                        nargs='+', type=int, default=[42])
    args = parser.parse_args()

    if not args.load_models or not os.path.isfile('hcv_igh_prepared.tsv'):
        create_trimmed_dataset('hcv_igh.tsv')
    immune2vec_model_hcv_100dim = create_model('hcv_igh_prepared.tsv', 100, args.load_models)
    immune2vec_model_hcv_10dim = create_model('hcv_igh_prepared.tsv', 10, args.load_models)

    if not args.load_models or not os.path.isfile('celiac_igh_prepared.tsv'):
        create_trimmed_dataset('celiac_igh.tsv')
    immune2vec_model_celiac_100dim = create_model('celiac_igh_prepared.tsv', 100, args.load_models)
    immune2vec_model_celiac_10dim = create_model('celiac_igh_prepared.tsv', 10, args.load_models)

    results = pd.DataFrame(columns=['random_seed', 'dataset', 'embedding', 'corpus', 'classifier', 'f1-score',
                                    'samples_per_family', 'best_params'])

    if args.dataset == DataSet.hcv or args.dataset == DataSet.all:
        print('starting hcv analysis')
        hcv_df = pd.read_csv('hcv_igh_prepared.tsv', sep='\t')
        results = test_immune2vec_100dim(hcv_df, immune2vec_model_hcv_100dim, 'hcv', 'hcv', 150000, results,
                                         args)
        results = test_immune2vec_10dim(hcv_df, immune2vec_model_hcv_10dim, 'hcv', 'hcv', 150000, results,
                                        args)
        results = test_immune2vec_100dim(hcv_df, immune2vec_model_celiac_100dim, 'celiac', 'hcv', 150000,
                                         results, args)
        results = test_immune2vec_10dim(hcv_df, immune2vec_model_celiac_10dim, 'celiac', 'hcv', 150000,
                                        results, args)
        results = test_atchely_factors(hcv_df, 'hcv', 150000, results, args)
        del hcv_df

    if args.dataset == DataSet.celiac or args.dataset == DataSet.all:
        print('starting celiac analysis')
        celiac_df = pd.read_csv('celiac_igh_prepared.tsv', sep='\t')
        results = test_immune2vec_100dim(celiac_df, immune2vec_model_hcv_100dim, 'hcv', 'celiac', 300000,
                                         results, args)
        results = test_immune2vec_10dim(celiac_df, immune2vec_model_hcv_10dim, 'hcv', 'celiac', 300000,
                                        results, args)
        results = test_immune2vec_100dim(celiac_df, immune2vec_model_celiac_100dim, 'celiac', 'celiac', 300000,
                                         results, args)
        results = test_immune2vec_10dim(celiac_df, immune2vec_model_celiac_10dim, 'celiac', 'celiac', 300000,
                                        results, args)
        results = test_atchely_factors(celiac_df, 'celiac', 300000, results, args)
        del celiac_df

    if args.dataset == DataSet.flu or args.dataset == DataSet.all:
        print('starting flu analysis')
        if not os.path.isfile('flu_igh_prepared.tsv'):
            flu_df = create_trimmed_dataset('flu_igh.tsv')
        else:
            flu_df = pd.read_csv('flu_igh_prepared.tsv', sep='\t')
        results = test_immune2vec_100dim(flu_df, immune2vec_model_hcv_100dim, 'hcv', 'flu', 100000,
                                         results, args)
        results = test_immune2vec_10dim(flu_df, immune2vec_model_hcv_10dim, 'hcv', 'flu', 100000,
                                        results, args)
        results = test_immune2vec_100dim(flu_df, immune2vec_model_celiac_100dim, 'celiac', 'flu', 100000,
                                         results, args)
        results = test_immune2vec_10dim(flu_df, immune2vec_model_celiac_10dim, 'celiac', 'flu', 100000,
                                        results, args)
        results = test_atchely_factors(flu_df, 'flu', 100000, results, args)
        del flu_df

    results.to_csv('results_' + time.strftime("%Y%m%d_%H%M%S") + '_ds_' + str(args.dataset).split('.')[1] + '_model_'
                   + str(args.classifier).split('.')[1] + '.csv', index=False)
    os.remove('results_tmp_pid' + str(os.getpid()) + '.csv')


if __name__ == '__main__':
    main()
