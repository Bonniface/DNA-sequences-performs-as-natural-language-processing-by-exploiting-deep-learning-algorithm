import sys
import os
import pandas as pd
from . import sequence_modeling
import argparse


def generate_model_add_args(parser):

    parser.add_argument('data_file_path', help='the input data file (tsv)', type=str)
    parser.add_argument('output_folder_path', help='Output folder where the model files will be created')
    parser.add_argument('desc', help='desc for the output files names', type=str)
    parser.add_argument('aa_col', help='name of the column with the amino acid sequence to be embedded, default is '
                                       '"cdr3_aa"', default="cdr3_aa")
    parser.add_argument('--n_dim', help='vector size (default 100)', default=100, type=int)
    parser.add_argument('--n_gram', help='n-gram parameter from the prot to vec (default 3)', type=int, default=3)
    parser.add_argument('--corpus_file', help='an input corpus file - if data-file was not provided', type=str)
    parser.add_argument('--reading_frame', help='reading frame parameter for the prot2vec', type=int)
    parser.add_argument('--data_fraction', help='fraction of the data used for creating the model (default 1.0)',
                        type=float, default=1.0)
    parser.add_argument('--seed', help='seed used for sampling the data for the model, (default 0)', type=int,
                        default=0)

    parser.set_defaults(func=generate_model_exec)


def generate_model_exec(args):

    aa_col = args.aa_col
    data = None
    if args.data_file_path is not None:
        if not os.path.isfile(args.data_file_path):
            print('Invalid data_file argument: {}\n Exiting...'.format(args.data_file))
            sys.exit(2)
        data_df = pd.read_csv(args.data_file_path, sep='\t')
        if aa_col not in data_df.columns:
            print('{} is not in {} columns\n Exiting...'.format(args.data_file_path))
            sys.exit(2)
        data = data_df[aa_col]
    elif args.corpus_file is not None:
        if not os.path.isfile(args.corpus_file):
            print('Invalid corpus_file argument: {}\n Exiting...'.format(args.corpus_file))
            sys.exit(2)
    else:
        print('Must provide a data file or a corpus file\n Exiting...'.format(args.corpus_file))
        sys.exit(2)

    pv = sequence_modeling.ProtVec(data=data, corpus=os.path.join(args.output_folder_path, args.desc),
                                   n=args.n_gram, reading_frame=args.reading_frame,
                                   size=args.n_dim, out=args.desc, sg=1, window=5, min_count=2, workers=3,
                                   sample_fraction=args.data_fraction, random_seed=args.seed)

    print('Model is ready, saving...')
    pv.save(os.path.join(args.output_folder_path, args.desc + '.model'))


def main():

    parser = argparse.ArgumentParser(description='generate prot2vec model from specified column in the data file')
    generate_model_add_args(parser)
    args = parser.parse_args()
    generate_model_exec(args)


if __name__ == '__main__':
    main()


