#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Extract features and learn from them, without saving in between.
'''

__author__ = "Johannes Bjerva, and Malvina Nissim"
__credits__ = ["Johannes Bjerva", "Malvina Nissim"]
__license__ = "GPL v3"
__version__ = "0.2"
__maintainer__ = "Mike Zhang"
__email__ = "mikz@itu.dk"
__status__ = "early alpha"

from feature_extractor import *
from learn_from_data import *
import logging
logging.basicConfig(format='%(levelname)s %(message)s', level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='feature csv filename', type=str, required=True)
    parser.add_argument('--fname', help='filename to store features', type=str, default=None)
    parser.add_argument('--nwords', type=int)
    parser.add_argument('--nchars', type=int)
    parser.add_argument('--split', help='Indicate what split the ML model has to use', type=str, nargs='+', default=None)
    parser.add_argument('--features', nargs='+', default=[])
    parser.add_argument('--dtype', help='datatype in file', type=str, default=None)  # TODO: Not implemented
    parser.add_argument('--delimiter', help='csv delimiter', type=str, default=',')  # TODO: Not implemented
    parser.add_argument('--lang', help='data language', type=str, default='english')

    parser.add_argument('--npz', help='feature npz filename', type=str)
    parser.add_argument('--algorithms', help='ml algorithms', nargs='+', required=True)
    parser.add_argument('--plot', help='Show plot', action='store_true')
    parser.add_argument('--cm', help='Show confusion matrix', action='store_true')
    parser.add_argument('--norm', help='Normalise confusion matrix', action='store_true')
    parser.add_argument('--min-samples', help='Min leaf samples in decision tree', type=int, default=1)
    parser.add_argument('--max-nodes', help='Max leaf nodes in decision tree', type=int, default=None)
    parser.add_argument('--k', help='number of neighbours for k-NN', type=int, default=1)
    parser.add_argument('--max-train-size', help='maximum number of training instances to look at', type=int, default=None)

    args = parser.parse_args()

    logging.info('Reading features...')
    X, y = read_features_from_csv(args)
    logging.info('Using one hot encoding...')
    X, feature_ids = features_to_one_hot(X)
    train_X, train_y, dev_X, dev_y, test_X, test_y = make_splits(X, y, args)

    if args.max_train_size:
        train_X = train_X[:args.max_train_size]
        train_y = train_y[:args.max_train_size]

    logging.info(f'There are {len(train_y)} train samples')
    logging.info(f'Classifier uses a {args.split[0]}% train and {args.split[1]}% test split.')
    baseline(train_y, dev_y)
    classifiers = get_classifiers(args)

    for clf in classifiers:
        clf.fit(train_X, train_y)
        training_result = evaluate_classifier(clf, train_X, train_y, args)
        dev_result = evaluate_classifier(clf, dev_X, dev_y, args)
        logging.info(f'Results on the train set:\n{training_result}\n')
        logging.info(f'Results on the dev set:\n{dev_result}')
