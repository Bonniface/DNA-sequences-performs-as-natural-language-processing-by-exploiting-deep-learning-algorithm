"""
Created on Sun Jul 22 14:32:57 2018

original @author: https://github.com/kyu999/biovec/blob/master/biovec/models.py
@author: miri-o

"""
from gensim.models import word2vec
from random import randint
import random
import numpy as np
import gensim


assert gensim.__version__ == '3.8.3'


def split_ngrams_with_repetition(seq, n, n_read_frames=None):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    """
    if n_read_frames is None or n_read_frames > n:
        n_read_frames = n

    frames = [zip(*[iter(seq[frame:])]*n) for frame in range(n_read_frames)]

    str_ngrams = []
    for ngrams in frames:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams


def split_ngrams_no_repetition(seq, n, reading_frame):
    """
    'acccgtgtctgg', n=3, reading frame = 1: ['acc', 'cgt', 'gtc', 'tgg']
    reading frame = 2: ['ccc', 'gtg', 'tct']
    reading frame = 3: ['ccg', 'tgt', 'ctg']
    """
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams[reading_frame-1]


def split_to_random_n_grams(seq, n):
    """
    n = a tuple (start, end) indicating the range of n's for random sampling 
    e.g. n = (3, 8) will split the sequence to n-grams with sizes ranging from 3-8
    'AGAMQSASMRDSRGPDPVSRATHNWFDP' : ['AGA', 'MQSASM', 'RDSRGPDPV', 'SRAT', 'HNWFDP']
    """
    str_ngrams = []
    current_n = randint(n[0], n[1])
    while len(seq)-n[0]>current_n:
        str_ngrams.append(str(seq[:current_n]))
        seq = seq[current_n:]
        current_n = randint(n[0], n[1])
    str_ngrams.append(str(seq))
    return str_ngrams


def generate_corpusfile(records, n, out, reading_frame = None, trim = None, sample_fraction=1.0, random_seed=5):
    '''
    Args:
        corpus_fname: corpus file name
        n: the number of chunks to split. In other words, "n" for "n-gram". 
        for a constant n splitting - n is an integer, for a random range, n should be a tuple of (start, end)
        reading_frame: 1/2/3 for splitting, default: None, including repetition (generating 3 overlaps)
        out: output corpus file path
        trim : typle (from start, drom end) - how many characteres to trim from the beginning and from the end
        portion: what portion of the sequences in the data will be used for the corpus
        seed: the random seed for randomly choosing the sequences for the corpus according to the portion parameter
    Description:
        Protvec uses word2vec inside, and it requires to load corpus file
        to generate corpus.
        
    '''
    f = open(out, "w")

    if sample_fraction != 1.0:
        random.seed(random_seed)
        records = np.random.choice(records, int(sample_fraction * len(records)), False)

    print('using {} records'.format(len(records)))
    for r in records:
        if trim:
            r = r[trim[0]:-trim[1]]
        if (reading_frame is None) and type(n) == int:
            ngram_patterns = split_ngrams_with_repetition(r, n)
        elif type(n) == int:
            ngram_patterns = split_ngrams_no_repetition(r, n, reading_frame)
        elif type(n) == tuple:
            ngram_patterns = split_to_random_n_grams(r, n)
        else:
            print('Error building corpus file, make sure n is and integer for contant n-grams, '
                  'or a tuple for random length n-grams')
            f.close()
            break
        if type(ngram_patterns[0])==list:
            for sub_seq in ngram_patterns:
                f.write(" ".join(sub_seq) + "\n")
        else:
            f.write(" ".join(ngram_patterns) + "\n")
    print('saved corpus file {}'.format(out))
    f.close()


def load_protvec(model_fname):
    return word2vec.Word2Vec.load(model_fname)


class ProtVec(word2vec.Word2Vec):

    def __init__(self, data=None, corpus=None, n=3, reading_frame=None, trim=None, size=100, out="prot2vec",
                 sg=1, window=25, min_count=2, workers=3, sample_fraction=1.0, random_seed=5):
        """
        Either fname or corpus is required.
        corpus_fname: data for corpus
        corpus: corpus object implemented by gensim
        n: n of n-gramp. single integer for a costant n, and a string ‘(start, end)’ for random splitting.
        reading frame : default None. possible values: 1/2/3/None – for all options
        trim: paramter for trimming the sequences, string format ‘(chars from start, chars from end)’ 
        out: corpus output file path
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
        portion: what portion of the sequences in the data will be used for the corpus
        seed: the random seed for randomly choosing the sequences for the corpus according to the portion parameter
        """

        self.n = n
        self.reading_frame = reading_frame
        self.size = size
        self.data = data
        self.trim = trim
        self.window = window

        if corpus is None and data is None:
            raise Exception("Either corpus_fname or corpus is needed!")

        if data is not None:
            print('Generate Corpus file from data...')
            generate_corpusfile(data, n, out + '_corpus.txt', reading_frame, trim, sample_fraction, random_seed)
            corpus = word2vec.Text8Corpus(out + '_corpus.txt')

        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)
        print('word2vec model, size={}, window={}, min_count={}, workers={})'.format(size, window, min_count, workers))

    def to_vecs(self, seq, n_read_frames=None):
            """
            convert sequence to three n-length vectors
            e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
            !!!FIX 06.11.2020:
            summarize the three vector to one using weighted average
            """
            ngram_patterns = split_ngrams_with_repetition(seq, self.n, n_read_frames)

            protvecs = []
            words_len = 0
            for ngrams in ngram_patterns:
                ngram_vecs = []
                for ngram in ngrams:
                    try:
                        ngram_vecs.append(self[ngram])
                    except:
                        raise KeyError("Model has never trained this n-gram: " + ngram)
                protvecs.append(sum(ngram_vecs))
                words_len += np.shape(ngram_vecs)[0]
            final_vec = sum(protvecs) / words_len
            return final_vec

