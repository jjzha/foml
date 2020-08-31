#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#This script reads a saved numpy array with features prepared for sklearn.
#The features are then used to learn something from the data.

__author__ = "Johannes Bjerva, and Malvina Nissim"
__credits__ = ["Johannes Bjerva", "Malvina Nissim"]
__license__ = "GPL v3"
__version__ = "0.3 (31/08/2020)"
__maintainer__ = "Mike Zhang"
__email__ = "mikz@itu.dk"
__status__ = "Testing"

import argparse
import logging
import random
from collections import Counter
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# random.seed(1337)
logging.basicConfig(format='%(levelname)s %(message)s', level=logging.DEBUG)


def read_features(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(fname, 'rb') as in_f:
        loaded = np.load(in_f)

        return loaded['X'], loaded['y']

def make_splits(X: np.ndarray, 
                y: np.ndarray, 
                args: argparse.Namespace) -> Tuple[List, List, List, List]:
    X: list = list(X)
    y: list = list(y)

    train: float = float(args.split[0])/100.0
    test: float = float(args.split[1])/100.0

    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)

    train_split = int(len(y) * train)
    dev_split = int((len(y) * train) * test)

    train_X = X[:train_split]
    train_y = y[:train_split]
    dev_X = train_X[:dev_split]
    dev_y = train_y[:dev_split]
    test_X  = X[train_split:]
    test_y  = y[train_split:]

    return train_X, train_y, dev_X, dev_y, test_X, test_y

def baseline(train_y: List[Union[int, str]], test_y: List[Union[int, str]]) -> None:
    most_common = Counter(train_y).most_common()[0][0]
    baseline = sum([1 for label in test_y if label == most_common]) / float(len(test_y))
    logging.info(f'Most frequent label: {most_common}')
    logging.info(f'Baseline accuracy: {baseline}')

def get_classifiers(args: argparse.Namespace) -> List[object]:
    classifiers = []

    if 'nb' in args.algorithms:
        classifiers.append(MultinomialNB())
    if 'dt' in args.algorithms:
        classifiers.append(DecisionTreeClassifier(
        random_state=0,
        criterion='entropy',
        min_samples_leaf=args.min_samples,
        max_leaf_nodes=args.max_nodes))
    if 'svm' in args.algorithms:
        classifiers.append(LinearSVC(max_iter=500,random_state=0))
    if 'knn' in args.algorithms:
        classifiers.append(KNeighborsClassifier(n_neighbors=args.k))

    return classifiers

def evaluate_classifier(clf: object, 
                        test_X: List[Union[int, str]], 
                        test_y: List[Union[int, str]], 
                        args: argparse.Namespace) -> None:
    preds = clf.predict(test_X)
    accuracy = accuracy_score(preds, test_y)

    if args.cm or args.plot:
        show_confusion_matrix(test_y, preds, args)

    return f'Accuracy: {accuracy}, classifier: {clf}'

def show_confusion_matrix(test_y, pred_y, args):
    cm = confusion_matrix(test_y, pred_y, labels=sorted(list(set(test_y))))

    if args.norm:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)

    logging.debug('Showing Confusion Matrix')
    if args.cm:
        print(f'\n{pd.DataFrame(cm, index=sorted(list(set(test_y))), columns=sorted(list(set(test_y))))}\n')
    if args.plot:
        from plotting import plot_confusion_matrix # Import here due to potential matplotlib issues
        plot_confusion_matrix(cm, test_y)

    print(classification_report(test_y, pred_y, labels=sorted(list(set(test_y)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', help='feature npz filename', type=str)
    parser.add_argument('--algorithms', help='ml algorithms', nargs='+', required=True)
    parser.add_argument('--plot', help='Show plot', action='store_true')
    parser.add_argument('--cm', help='Show confusion matrix', action='store_true')
    parser.add_argument('--norm', help='Normalise confusion matrix', action='store_true')
    args = parser.parse_args()

    X, y = read_features(args.npz)
    train_X, train_y, dev_X, dev_y, test_X, test_y = make_splits(X, y, args)
    baseline(train_y, test_y)
    classifiers = get_classifiers(args)

    for clf in classifiers:
        clf.fit(train_X, train_y)
        evaluate_classifier(clf, test_X, test_y, args)
