from gensim.models import KeyedVectors
import glob
import json
from matrixserializer import load_matrix
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from ioutils import load_pickle, mkdir


def load_coha():
    vectors_list = glob.glob('../../Replication-Garg-2018/data/coha-word/*vectors.txt')
    vectors = {}
    for file_name in vectors_list:
        file_decade = file_name.split(os.path.sep)[-1][:4]
        vectors[file_decade] = KeyedVectors.load_word2vec_format(file_name, binary=False, no_header=True)
    return vectors

# Parameters
K = 5  # HistWords indicate they follow Levy et al. (2015) in order to set hyperparameters.
# Seems like k is 5 per https://aclanthology.org/Q15-1016.pdf but unsure

# Paths
bin_dir = '../cooccurs/word/4'
word_dict_pkl = '../info/word-dict.pkl'
output_path = '../results/PMI'

mkdir(output_path)

# Load word dictionary and word lists
word_dict = load_pickle(word_dict_pkl)
with open('../../Local/word_lists/word_lists_all.json', 'r') as file:
    word_list_all = json.load(file)
vectors = load_coha()

# Add stopwords
word_list_all['Stopwords'] = stopwords.words('english')


def get_word_df(w_dict, m, w=None, w_idx=None):
    if w_idx is None:
        assert w is not None
        try:
            w_idx = w_dict[w]
        except KeyError:
            return None, None

    try:
        w_coo = m[w_idx].tocoo()
        w_df = pd.DataFrame({'w_idx': w_coo.row, 'c_idx': w_coo.col, '#wc': w_coo.data})
    except IndexError:
        return None, None

    w_sum = w_df['#wc'].sum()
    return w_df, w_sum


def dot_product(w, c, vec):
    try:
        w_idx_v = vec.key_to_index[w]
        c_idx_v = vec.key_to_index[c]
    except KeyError:
        return None

    v_w = vec.vectors[w_idx_v]
    v_c = vec.vectors[c_idx_v]

    if np.linalg.norm(v_w) < 1e-6 or np.linalg.norm(v_c) < 1e-6:
        return None

    return np.dot(v_w, v_c)


def idx_to_word(idx, w_dict):
    w = list(w_dict.keys())[idx]
    assert idx == w_dict[w]
    return w


# Get surname metrics for each decade
# CSR matrix is in the format (word, context)
def process_matrix(m, w_dict, word_lists, v):
    wls = ['Asian_San_Bruno_All', 'White_San_Bruno_All', 'Otherization Words',
           'PNAS Asian Target Words', 'PNAS White Target Words',
           'Stopwords']
    #wls = ['Asian_San_Bruno_All', 'White_San_Bruno_All', 'Otherization Words',
    #       'PNAS Asian Target Words', 'PNAS White Target Words']
    m_df = pd.DataFrame()
    for wl in tqdm(wls):
        for w in word_lists[wl]:
            w_df, w_sum = get_word_df(w=w, w_dict=w_dict, m=m)
            if w_df is None:
                continue

            # Get #(w,c), #(w), #(c)
            w_df['w'] = w
            w_df['w_idx'] = w_df['w'].apply(lambda word: w_dict[word])
            w_df['c'] = w_df['c_idx'].apply(lambda idx: idx_to_word(idx, w_dict))
            w_df['#w'] = w_sum
            w_df['#c'] = w_df['c_idx'].apply(lambda idx: get_word_df(w_idx=idx, w_dict=w_dict, m=m)[1])

            # Get w.c (note: we check that ||w||2 and ||c||2 >= 1e-6)
            w_df['w_dot_c'] = w_df.apply(
                lambda row: dot_product(w=row['w'], c=row['c'], vec=v), axis=1)

            # Append
            w_df['word_list'] = wl
            m_df = pd.concat([m_df, w_df])

    # Add |D|
    m_df['D'] = m.sum()

    return m_df


# Process each matrix
bin_files = glob.glob(os.path.join(bin_dir, '*.bin'))

pmi_df = pd.DataFrame()
if os.path.exists(os.path.join(output_path, 'pmi.csv')):
    pmi_df = pd.read_csv(os.path.join(output_path, 'pmi.csv'))

for bin_file in tqdm(bin_files):
    decade = bin_file.split(os.path.sep)[-1].replace('.bin', '')

    if 'decade' in pmi_df.columns and int(decade) in list(pmi_df['decade'].unique()):
        continue
    matrix = load_matrix(bin_file)
    matrix_df = process_matrix(m=matrix, w_dict=word_dict, word_lists=word_list_all, v=vectors[decade])
    matrix_df['decade'] = int(decade)

    # Compute PMI
    matrix_df['PMI'] = matrix_df.apply(
        lambda row: np.log((row['#wc'] * row['D']) / (row['#w'] * row['#c'])) - np.log(K) if row['#w'] * row[
            '#c'] != 0 else None, axis=1)

    pmi_df = pd.concat([pmi_df, matrix_df])

    # Save output
    pmi_df.to_csv(os.path.join(output_path, 'pmi.csv'), index=False)
