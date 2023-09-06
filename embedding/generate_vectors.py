import pandas as pd
import numpy as np
import os
import sys
import argparse
from . import sequence_modeling


def generate_vectors_add_args(subparsers):

    parser = subparsers.add_parser('generate_vectors', description='embed a column from input file')
    parser.add_argument('data_file_path', help='a tsv with the column to vectorize')
    parser.add_argument('model_file_path', help='a saved word embedding model file')
    parser.add_argument('output_folder_path', help='Output folder where the model files will be created')
    parser.add_argument('--column', help='column name to convert to vectors (default cdr3_aa)', default="cdr3_aa")
    parser.add_argument('--n_read_frames', help='how many read frames to use for the embedding, default is all possible'
                                                'read frames', type=int)

    parser.set_defaults(func=generate_vectors_exec)


def generate_vectors_exec(args):

    data_file_path = args.data_file_path
    model_file_path = args.model_file_path
    column = args.column
    n_read_frames = args.n_read_frames

    if not os.path.isfile(data_file_path):
        print('Data file ({}) error! Make sure the file exists and it is *.tsv file. Exiting...'.format(data_file_path))
        sys.exit(1)
    print('Input file for embedding: ', data_file_path, '\nModel: ', model_file_path)
 
    data_file = pd.read_csv(data_file_path, sep='\t')

    # load saved model
    model = sequence_modeling.load_protvec(model_file_path)

    # generate a vector for each junction
    data_len = len(data_file)
    print('Data length: ' + str(data_len))

    def embed_data(word):
        try:
            return list(model.to_vecs(word, n_read_frames=n_read_frames))
        except:
            return np.nan

    print('Generating vectors...')
    vectors = data_file[column].apply(embed_data)

    print('{:.3}% of data not transformed'.format((100*sum(vectors.isna())/data_len)))

    # drop the un translated rows from the file
    data_file = data_file.drop(vectors[vectors.isna()].index, axis=0)
    vectors = vectors[vectors.notna()]

    # save to files:
    data_file_name = os.path.basename(data_file_path).split(".tsv")[0]
    model_file_name = os.path.basename(model_file_path).split(".model")[0]

    data_file_output = os.path.join(args.output_folder_path, data_file_name + '_filtered.tsv')
    print('Saving ' + data_file_output)
    data_file.to_csv(data_file_output, sep='\t', index=False)

    vectors = np.array(vectors.tolist())

    vectors_file_output = os.path.join(args.output_folder_path, data_file_name + '_' + model_file_name + '_vectors.npy')
    print('Saving ' + vectors_file_output)
    np.save(vectors_file_output, vectors)


def main():

    parser = argparse.ArgumentParser(description='generate prot2vec model from specified column in the data file')
    generate_vectors_add_args(parser)
    args = parser.parse_args()
    generate_vectors_exec(args)


if __name__ == '__main__':
    main()

