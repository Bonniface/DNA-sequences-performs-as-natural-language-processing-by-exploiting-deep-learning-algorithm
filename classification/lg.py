from . import common
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import argparse
from enum import Enum


class Penalty(Enum):
    l2 = 1
    elasticnet = 2


def lg_add_args(parser):

    parser.add_argument('--penalty', help='The penalty to use for the lg regularization. Default is l2.',
                        type=lambda x: Penalty.__members__.get(x).value, default=Penalty.l2.value)
    parser.add_argument('--optimize',
                        help='Use grid search with cross validation to search optimized regularization '
                             'hyper parameters.',
                        type=lambda x: x.lower() in ("yes", "true", "t", "1"), default=False)
    parser.add_argument('--C', help='Inverse of regularization strength - a positive float, default is 1.0. Smaller '
                                    'values specify stronger regularization.', type=float, default=1.0)
    parser.add_argument('--l1_ratio',
                        help="The list of Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only relevant if "
                             "penalty='elasticnet'.",
                        type=float, default=0.5)

    common.add_common_arguments(parser)
    parser.set_defaults(func=lg_exec)


def lg_exec(args):

    train_features, test_features, train_labels, test_labels, labels_to_classify = common.read_common_args(args)

    if args.optimize is True:
        if args.penalty == Penalty.l2.value:
            clf = LogisticRegressionCV(Cs=np.arange(0.01, 0.5, 0.01), penalty='l2', solver='liblinear', random_state=0,
                                       class_weight='balanced', max_iter=100, refit=False)
        else:
            clf = LogisticRegressionCV(l1_ratios=np.arange(0.01, 1.0, 0.01), penalty='elasticnet', solver='saga',
                                       class_weight='balanced', random_state=0, max_iter=100, refit=False)
    else:
        if args.penalty == Penalty.l2.value:
            clf = LogisticRegression(C=args.C, class_weight='balanced', penalty='l2', solver='liblinear')
        else:
            clf = LogisticRegression(l1_ratio=args.l1_ratio, penalty='elasticnet', solver='saga')

    return common.classify(train_features, test_features, train_labels, test_labels, clf, labels_to_classify)


def main():

    parser = argparse.ArgumentParser(description="Logistic regression classifier")
    lg_add_args(parser)
    args = parser.parse_args()
    parser.func(args)


if __name__ == '__main__':
    main()


