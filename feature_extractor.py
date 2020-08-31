#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#This script reads a CSV file and extracts pre-defined features from it.
#The features are saved in a scikit-learn-friendly manner.

__author__ = "Johannes Bjerva, and Malvina Nissim"
__credits__ = ["Johannes Bjerva", "Malvina Nissim"]
__license__ = "GPL v3"
__version__ = "0.3 (31/08/2020)"
__maintainer__ = "Mike Zhang"
__email__ = "mikz@itu.dk"
__status__ = "early alpha"

import argparse
import csv
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(format='%(levelname)s %(message)s', level=logging.INFO)
label_to_id: defaultdict = defaultdict(lambda: len(label_to_id))
cat_to_id: defaultdict = defaultdict(lambda: len(cat_to_id))

def read_features_from_csv(args: argparse.Namespace) -> Tuple[List, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[np.ndarray] = []

    with open(file=args.csv, mode='r', encoding='utf8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=args.delimiter)
        header: List[str] = next(csv_reader)
        label_index: int = header.index('label')

        try:
            if args.features is not None:
                for feature in args.features:
                    if feature in header and 'text' in feature:
                        text_index = header.index(feature)
                    else:
                        text_index = -1
            else:
                text_index = header.index('text-cat')
        except:
            logging.warning('No text-cat found')
            text_index = -1

        feature_indices: List[int] = []

        for feature in args.features:
            if text_index >= 0:
                pass
            elif feature in header:
                feature_indices.append(header.index(feature))
            else:
                logging.warning(f'Feature {feature} not found in header')

        types = get_column_types(header)

        for line in csv_reader:
            label, features = get_line_features(line, types, label_index, text_index, feature_indices, args)
            #TODO: Get numerical features
            X.append(features)
            y.append(label)

    return X, np.asarray(y, dtype=str)

def get_column_types(header: List[str]) -> np.ndarray:
    types: np.ndarray = np.zeros((len(header), ), dtype=np.object)

    for idx, name in enumerate(header):
        if 'cat' in name:
            types[idx] = np.ndarray # TODO
        else:
            types[idx] = np.float32

    return types

def get_line_features(line: List[str], 
                      feature_dtypes: np.ndarray, 
                      label_index: int, 
                      text_index: int, 
                      feature_indices: List[int], 
                      args: argparse.Namespace) -> Tuple[List[int], np.ndarray]:
    '''Gets the features in a line.Must have the format (label, feature(s)).'''
    #TODO: Add error handling / messages
    # Could go wrong:
    # * Not all features defined
    # * Some features need to be converted to categories
    # * Lemmatisation etc. for text
    label: List[int] = line[label_index]
    features: List[str] = []

    for idx, column in enumerate(line):
        if idx == label_index: 
            continue
        if idx in feature_indices:
            #TODO: Fix non-categorical
            features.append(cat_to_id[column+'idx'])

        elif idx == text_index:
            sentence_features = []

            if args.nwords:
                for n in range(args.nwords):
                    ngrams = find_ngrams(column.split(), n+1)
                    sentence_features.extend([cat_to_id[' '.join(ngram)] for ngram in ngrams])

            if args.nchars:
                for n in range(args.nchars):
                    ngrams = find_ngrams(' '.join(column.split()), n+1)
                    sentence_features.extend([cat_to_id[' '.join(ngram)] for ngram in ngrams])

            features.extend(sentence_features)

    features = np.asarray(features)

    return label, features

def find_ngrams(sentence: str, n: int) -> List[Tuple]:
    return list(zip(*[sentence[idx:] for idx in range(n)]))

def preprocess(word: str) -> str:
    return word.strip()

def features_to_one_hot(X: List[np.ndarray]) -> Tuple[np.ndarray, Dict]:
    '''Convert, e.g., word id features to one hot representation'''

    feature_counts: Counter = Counter([i for j in X for i in j])
    count_cutoff: int = int(len(X) * 0.001)
    features_to_use: set = set([feature for feature, count in feature_counts.items() if count > count_cutoff])
    new_feature_ids: defaultdict = defaultdict(lambda: len(new_feature_ids))

    for feature in features_to_use:
        new_feature_ids[feature]

    n_cats = len(new_feature_ids)
    logging.info(f'Number of features: {n_cats}')
    one_hot_X = np.zeros((len(X), n_cats), dtype=np.float32)
    # TODO: Fix for several cats

    for idx, sentence in enumerate(X):
        for cat_id in sentence:
            if cat_id in features_to_use:
                one_hot_X[idx, new_feature_ids[cat_id]] = 1

    one_hot_X /= np.max(one_hot_X, axis=0)

    id_to_cat = dict([(idx, cat) for cat, idx in cat_to_id.items()])
    id_to_char = dict([(new_id, id_to_cat[old_id]) for old_id, new_id in new_feature_ids.items()])

    return one_hot_X, id_to_char

def save_features(X: np.ndarray, y: np.ndarray, fname: str) -> None:
    '''Save X and y to a compressed .npz file'''
    np.savez_compressed(fname, X=X, y=y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='feature csv filename', type=str, required=True)
    parser.add_argument('--fname', help='filename to store features', type=str, default=None)
    parser.add_argument('--nwords', type=int)
    parser.add_argument('--nchars', type=int)
    parser.add_argument('--features', nargs='+', default=[])
    parser.add_argument('--dtype', help='datatype in file', type=str, default=None)  # TODO: Not implemented
    parser.add_argument('--delimiter', help='csv delimiter', type=str, default=',')  # TODO: Not implemented
    parser.add_argument('--lang', help='data language', type=str, default='english')
    args = parser.parse_args()

    fname = args.csv[:-4] if not args.fname else args.fname

    logging.info('reading features...')
    X, y = read_features_from_csv(args)
    logging.info('one hot encoding...')
    X, feature_ids  = features_to_one_hot(X)
    logging.info(f'saving features to {fname}')
    save_features(X, y, fname)
