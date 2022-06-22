import numpy as np
import copy
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import warnings
import pdb

from methods_util import conformalize_scores

class BinaryConformal:
    def __init__(self, X_in, X_out, bbox, n_folds=10, random_state=2022, verbose=True):
        self.verbose = verbose
        self.n_folds = n_folds
        self.cv = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        self.bboxes = [copy.deepcopy(bbox) for k in range(self.n_folds)]

        # Split the inlier data into training and calibration subsets
        self.folds = [(train_idx, cal_idx) for train_idx, cal_idx in self.cv.split(X_in)]

        # Train and calibrate the model
        self._train_calibrate(X_in, X_out)


    def _train_calibrate(self, X_in, X_out):
        self.n_cal = X_in.shape[0]
        # Train the binary classification models
        self.models = [None] * self.n_folds
        k = 0
        for fold in self.folds:
            train_idx = fold[0]
            n_in_train = len(train_idx)
            n_out = X_out.shape[0]
            X_train = np.concatenate([X_in[train_idx], X_out],0)
            Y_train = np.concatenate([[0]*n_in_train, [1]*n_out])
            if self.verbose:
                print("Fitting the black-box model {:d} on {:d} inliers and {:d} outliers... ".format(k, n_in_train, n_out), end="")
                sys.stdout.flush()
            try:
                self.bboxes[k].fit(X_train, Y_train)
            except:
                print("Warning: cannot train binary classifier!")
            if self.verbose:
                print("done.")
                sys.stdout.flush()
            k += 1

        # Evaluate conformity scores on the hold-out inliers
        self.scores_cal = -np.ones((self.n_cal,))
        k = 0
        for fold in self.folds:
            cal_idx = fold[1]
            try:
                self.scores_cal[cal_idx] = self.bboxes[k].predict_proba(X_in[cal_idx])[:,0]
            except:
                self.scores_cal[cal_idx] = 1
            self.scores_cal[cal_idx] += np.random.normal(loc=0, scale=1e-6, size=(len(cal_idx),))
            k += 1

    def compute_pvalues(self, X_test):
        n_test = X_test.shape[0]
        scores_test = -np.ones((n_test,self.n_cal))
        k = 0
        for fold in self.folds:
            cal_idx = fold[1]
            try:
                scores_tmp = self.bboxes[k].predict_proba(X_test)[:,0]
            except:
                scores_tmp = 1
            scores_tmp += np.random.normal(loc=0, scale=1e-6, size=(len(scores_tmp),))
            scores_test[:,cal_idx] = np.tile(scores_tmp, (len(cal_idx),1)).T
            k += 1

        scores_cal_mat = np.tile(self.scores_cal, (n_test,1))
        pvals = (1.0+np.sum(scores_cal_mat <= scores_test,1))/(self.n_cal+1.0)
        return pvals

class OneClassConformal:
    def __init__(self, X_in, bbox, n_folds=10, random_state=2022, verbose=True):
        self.verbose = verbose
        self.n_folds = n_folds
        self.cv = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        self.bboxes = [copy.deepcopy(bbox) for k in range(self.n_folds)]

        # Split data into training and calibration subsets
        self.folds = [(train_idx, cal_idx) for train_idx, cal_idx in self.cv.split(X_in)]

        # Train and calibrate the model
        self._train_calibrate(X_in)


    def _train_calibrate(self, X_in):
        self.n_cal = X_in.shape[0]
        # Train the binary classification models
        self.models = [None] * self.n_folds
        k = 0
        for fold in self.folds:
            train_idx = fold[0]
            n_in_train = len(train_idx)
            if self.verbose:
                print("Fitting the black-box model {:d} on {:d} inliers... ".format(k, n_in_train), end="")
                sys.stdout.flush()
            try:
                self.bboxes[k].fit(X_in[train_idx])
            except:
                print("Warning: cannot train OCC!")
            if self.verbose:
                print("done.")
                sys.stdout.flush()
            k += 1

        # Evaluate conformity scores on the hold-out inliers
        self.scores_cal = -np.ones((self.n_cal,))
        k = 0
        for fold in self.folds:
            cal_idx = fold[1]
            try:
                self.scores_cal[cal_idx] = self.bboxes[k].score_samples(X_in[cal_idx])
            except:
                self.scores_cal[cal_idx] = 1
            self.scores_cal[cal_idx] += np.random.normal(loc=0, scale=1e-6, size=(len(cal_idx),))
            k += 1

    def compute_pvalues(self, X_test):
        n_test = X_test.shape[0]
        scores_test = -np.ones((n_test,self.n_cal))
        k = 0
        for fold in self.folds:
            cal_idx = fold[1]
            try:
                scores_tmp = self.bboxes[k].score_samples(X_test)
            except:
                scores_tmp = 1
            scores_tmp += np.random.normal(loc=0, scale=1e-6, size=(len(scores_tmp),))
            scores_test[:,cal_idx] = np.tile(scores_tmp, (len(cal_idx),1)).T
            k += 1

        scores_cal_mat = np.tile(self.scores_cal, (n_test,1))
        pvals = (1.0+np.sum(scores_cal_mat <= scores_test,1))/(self.n_cal+1.0)
        return pvals
