import changeo.Gene
import pandas as pd
import os
import numpy as np
from Bio.Seq import Seq
import argparse


def prepare_dataset_add_args(parser):

    parser.add_argument('data_file_path',
                        help='The preprocessed data file path')
    parser.add_argument('output_desc',
                        help='Description prefix for the test/train files name')
    parser.add_argument('output_folder_path',
                        help='Output folder to save output files')
    parser.add_argument('--is_naive',
                        help='Is that data from naive cells sequencing, default: "False"',
                        type=lambda x: x.lower() in ("yes", "true", "t", "1"), default=False)
    parser.add_argument('--min_seq_per_subject',
                        help='minimal number of sequences per subject, default: "2000"',
                        type=int, default=2000)
    parser.add_argument('--subject_col',
                        help='name of the column with the subject identifier, default: "subject_id"',
                        default="subject_id")
    parser.add_argument('--label_col',
                        help='name of the column with the labeling, default: "disease_diagnosis"',
                        default="disease_diagnosis")

    parser.set_defaults(func=prepare_dataset_exec)


def prepare_dataset_exec(args):

    output_desc = args.output_desc
    data_file_path = args.data_file_path
    is_naive = args.is_naive
    subject_col = args.subject_col
    min_seq_per_subject = args.min_seq_per_subject
    output_dir = args.output_folder_path
    label_col = args.label_col

    if not os.path.isfile(data_file_path):
        raise Exception('Data file error, make sure data file path: {}'.format(data_file_path))

    print('Loading data file {}...'.format(data_file_path))
    data_file = pd.read_csv(data_file_path, sep='\t')

    print('Load complete, preparing data...')
    prepared_data_file = prepare_data_file(data_file, is_naive, subject_col, label_col, min_seq_per_subject)

    output_filename = os.path.join(output_dir, output_desc + '_prepared.tsv')
    prepared_data_file.to_csv(output_filename, sep='\t', index=False)

    print('output saved to {}'.format(output_filename))


def prepare_data_file(data_file: pd.DataFrame, is_naive: bool, subject_col: str, label_col: str,
                      min_seq_per_subject: int) -> pd.DataFrame:

    required_columns = ['sequence_id', 'cdr3', 'v_call', 'j_call', 'productive', 'consensus_count',
                        subject_col, label_col]
    for column in required_columns:
        if column not in data_file.columns:
            raise Exception('"{}" column is not in data_file columns'.format(column))

    data_file = data_file[data_file['cdr3'].notnull()]
    data_file = data_file[data_file['cdr3'].str.find('.') < 0]
    data_file = data_file[data_file['cdr3'].str.find('-') < 0]
    data_file = data_file[data_file['v_call'].notnull()]
    data_file = data_file[data_file['j_call'].notnull()]

    if 'cdr3_aa' not in data_file.columns:
        data_file = data_file[data_file['cdr3'].str.len() % 3 == 0]
        data_file['cdr3_aa'] = data_file['cdr3'].apply(lambda x: Seq(x).translate().__str__())
        data_file = data_file[data_file['cdr3_aa'].str.find('*') < 0]

    if not np.issubdtype(data_file['consensus_count'].dtype, np.number):
        data_file.loc[data_file['consensus_count'].apply(str).str.contains('='), 'consensus_count'] = data_file.loc[
            data_file['consensus_count'].apply(str).str.contains('='), 'consensus_count'].apply(str).str.split(
            '=', 1).str[1].to_numpy().astype(int)

    data_file['cdr3_length'] = data_file['cdr3'].str.len()
    data_file['cdr3_aa_length'] = data_file['cdr3_aa'].str.len()

    data_file['v_gene'] = data_file['v_call'].apply(lambda x: changeo.Gene.getGene(x, action='first'))
    data_file['j_gene'] = data_file['j_call'].apply(lambda x: changeo.Gene.getGene(x, action='first'))

    if (subject_col != 'subject_id') and (subject_col in data_file.columns) and ('subject_id' in data_file.columns):
        data_file.drop(columns=['subject_id'], inplace=True)

    if (label_col != 'disease_diagnosis') and (label_col in data_file.columns) and ('disease_diagnosis' in data_file.columns):
        data_file.drop(columns=['disease_diagnosis'], inplace=True)

    data_file.rename(columns={subject_col: 'subject_id', label_col: 'disease_diagnosis'}, inplace=True)

    by_subject = data_file.groupby(['subject_id'])

    df_list = []

    total_none_productive = 0
    total_too_short_cdr3 = 0
    total_conscount_is_one = 0
    total_too_many_v_mutations = 0

    for subject, frame in by_subject:

        subject_frame = frame

        cur_len = len(subject_frame)

        print('subject {} total rows: {} label'.format(subject, len(subject_frame), frame['disease_diagnosis'].iloc[0]))

        # 1. REMOVE NON-FUNCTIONAL SEQUENCES
        subject_frame = subject_frame.loc[subject_frame.productive.apply(str).str.lower().isin(
            ['t', 'true', '1', 'yes']), :]
        print(' - After removing non functional sequences: {} ({}%)'.format(
            len(subject_frame), len(subject_frame)*100/cur_len))
        total_none_productive += (cur_len-len(subject_frame))
        cur_len = len(subject_frame)

        # 2. REMOVE CDR3 shorter equal to 3
        subject_frame = subject_frame[subject_frame.cdr3_aa_length > 3]
        print(' - After cdr3_aa_length > 3 {} ({}%)'.format(
            len(subject_frame), len(subject_frame)*100/cur_len))
        total_too_short_cdr3 += (cur_len-len(subject_frame))
        cur_len = len(subject_frame)

        # 3. LEAVE ONLY ROWS WHERE CONSCOUNT > 1
        subject_frame = subject_frame[subject_frame['consensus_count'] > 1]
        print(' - After consensus_count > 1: {} ({}%)'.format(
            len(subject_frame), len(subject_frame)*100/cur_len))
        total_conscount_is_one += (cur_len-len(subject_frame))
        cur_len = len(subject_frame)

        # 4. FOR NAIVE CELLS ONLY ALLOW 3 MUTATIONS DIFFERENCE FROM THE GERMLINE
        if (is_naive and ('cdr_r' in subject_frame.columns) and ('cdr_s' in subject_frame.columns) and
            ('fwr_r' in subject_frame.columns) and ('fwr_s' in subject_frame.columns)):
            tot_mutations_cnt = (subject_frame['cdr_r'] + subject_frame['cdr_s'] + subject_frame['fwr_r'] +
                                 subject_frame['fwr_s'])
            subject_frame = subject_frame[tot_mutations_cnt <= 3]
            print(' - After v_mutations_count <= 3: {} ({}%)'.format(
                len(subject_frame), len(subject_frame)*100/cur_len))
            total_too_many_v_mutations += (cur_len - len(subject_frame))
            cur_len = len(subject_frame)

        if len(subject_frame) < min_seq_per_subject:
            print('subject not added due to low number of sequences')
            continue

        if subject_frame['cdr3_aa'].value_counts(normalize=True)[0] > 0.5:
            print('subject not added because more than 50% of the sequences are identical')
            continue

        subject_frame = subject_frame.loc[:, ['sequence_id', 'subject_id', 'disease_diagnosis',
                                              'v_gene', 'j_gene', 'v_call', 'j_call', 'cdr3_aa', 'cdr3_aa_length']]
        df_list += [subject_frame]

    big_df = pd.concat(df_list)

    print('total none productive discarded: {}'.format(total_none_productive))
    print('total too short cdr3 discarded: {}'.format(total_too_short_cdr3))
    print('total conscount=1 discarded: {}'.format(total_conscount_is_one))
    if is_naive is True:
        print('total too many v mutations discarded: {}'.format(total_too_many_v_mutations))

    subjects = big_df.loc[:, ].groupby(by=['subject_id'])['disease_diagnosis'].apply(lambda x: x.iloc[0])
    print('Summary:')
    print('Subjects (total {}):\n {}\n {}\n'.format(len(subjects), subjects.value_counts(), subjects.value_counts(
        normalize=True)))
    print('Sequences (total {}):\n {}\n {}\n'.format(len(big_df), big_df['disease_diagnosis'].value_counts(),
                                      big_df['disease_diagnosis'].value_counts(normalize=True)))
    for label in big_df['disease_diagnosis'].unique():
        print('label {} avg sequences per subject {}'.format(
            label, sum(big_df['disease_diagnosis'] == label) / sum(subjects == label)))

    return big_df


def main():

    parser = argparse.ArgumentParser('prepare_dataset',
                                     description='Prepare tsv repertoires data file for the clustering steps. Sequences'
                                                 ' will be filtered according to:'
                                                 '1. Have cdr3_aa sequence and functional'
                                                 '2. Conscount > 1'
                                                 '3. If data is naive then v_gene and j_gene distance from germline '
                                                 '   must not exceed 3 mutations'
                                                 '4. Sequences belong to to a subject with less than minimal number '
                                                 '   will be discarded.'
                                                 '5. Sequences belong to a subject which more than 50% of its'
                                                 '   repertoire is the same sequence will be discarded.'
                                                 'Before clustering the file must include the following columns '
                                                 '(self explained):'
                                                 '["disease_diagnosis", "subject_id", "v_gene", "j_gene", "cdr3_aa", '
                                                 '"cdr3_aa_length"]')
    prepare_dataset_add_args(parser)
    args = parser.parse_args()
    prepare_dataset_exec(args)


if __name__ == '__main__':
    main()
