import numpy as np
import copy
import sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb

sys.path.append('../third_party')

class WeightedOneClassConformal:
    def __init__(self, X_in, X_out, bbox_in, bbox_out, calib_size=0.5, random_state=2022, tuning=True, verbose=True, progress=True):
        self.bbox_in = copy.deepcopy(bbox_in)
        self.bbox_out = copy.deepcopy(bbox_out)
        self.tuning = True
        self.verbose = verbose
        self.progress = progress

        # Split data into training and calibration subsets
        X_in_train, self.X_in_calib = train_test_split(X_in, test_size=calib_size, random_state=random_state)
        X_out_train, self.X_out_calib = train_test_split(X_out, test_size=calib_size, random_state=random_state)

        # Train the two black-box models
        self._train(self.bbox_in, X_in_train)
        self._train(self.bbox_out, X_out_train)

        # Pre-compute conformity scores for inlier calibration data using inlier model
        self.scores_in_calib = self.bbox_in.score_samples(self.X_in_calib)
        # Pre-compute conformity scores for inlier calibration data using outlier model
        self.scores_outin_calib = self.bbox_out.score_samples(self.X_in_calib)
        # Pre-compute conformity scores for outlier calibration data using outlier model
        self.scores_out_calib = self.bbox_out.score_samples(self.X_out_calib)
        # Pre-compute conformity scores for outlier calibration data using inlier model
        self.scores_inout_calib = self.bbox_in.score_samples(self.X_out_calib)

    def _train(self, bbox, X_train):
        # Fit the black-box one-class classification model on the training data
        if self.verbose:
            print("Fitting the black-box model on {:d} data points... ".format(X_train.shape[0]), end="")
            sys.stdout.flush()
        bbox.fit(X_train)
        if self.verbose:
            print("done.")
            sys.stdout.flush()

    def _calibrate_in(self, score_test):
        scores_out = self.scores_inout_calib
        # Concatenate conformity scores for inlier calibration data and test point
        scores = np.append(self.scores_in_calib, score_test)

        if self.tuning:
            # Large score <-> large p-value
            # We expect the outliers should have smaller scores
            median_out = np.median(scores_out)
            median_in = np.median(scores)
            if median_in < median_out:
                scores = -scores

        # Compute conformal p-values
        n_cal = len(scores) - 1
        scores_mat = np.tile(scores, (len(scores),1))
        tmp = np.sum(scores_mat <= scores.reshape(len(scores),1), 1)
        pvals = (1.0+tmp)/(1.0+n_cal)
        return pvals

    def _calibrate_out(self, score_test):
        # Concatenate conformity scores based on outlier model for inlier calibration data and test point
        scores = np.append(self.scores_outin_calib, score_test)
        scores_cal = self.scores_out_calib

        if self.tuning:
            # Large score <-> large p-value
            # We expect the inliers should haeve smaller scores
            median_out = np.median(scores_cal)
            median_in = np.median(scores)
            if median_in > median_out:
                scores = -scores
                scores_cal = -scores_cal

        # Compute conformal p-values
        n_cal = len(scores) - 1
        scores_mat = np.tile(scores_cal, (len(scores),1))
        tmp = np.sum(scores_mat <= scores.reshape(len(scores),1), 1)
        pvals = (1.0+tmp)/(1.0+n_cal)
        return pvals

    def compute_pvalues(self, X_test):
        # Compute conformity scores for test data
        scores_in_test = self.bbox_in.score_samples(X_test)
        scores_out_test = self.bbox_out.score_samples(X_test)

        def compute_pvalue(score_in_test, score_out_test):
            pvals_0 = self._calibrate_in(score_in_test)
            pvals_1 = self._calibrate_out(score_out_test)
            scores = pvals_0 / pvals_1
            # Compute final conformal p-value
            n_cal = len(scores) - 1
            scores_mat = np.tile(scores, (len(scores),1))
            tmp = np.sum(scores_mat <= scores.reshape(len(scores),1), 1)
            pvals = (1.0+tmp)/(1.0+n_cal)
            return pvals[-1]

        n_test = X_test.shape[0]
        if self.progress:
            iterator = tqdm(range(n_test))
        else:
            iterator = range(n_test)

        pvals = -np.ones((n_test,))
        for i in iterator:
            pvals[i] = compute_pvalue(scores_in_test[i], scores_out_test[i])

        return pvals


class OneClassConformal:
    def __init__(self, X_in, bbox, calib_size=0.5, random_state=2022, verbose=True):
        self.bbox = copy.deepcopy(bbox)
        self.verbose = verbose

        # Split data into training and calibration subsets
        X_train, X_calib = train_test_split(X_in, test_size=calib_size, random_state=random_state)

        # Train and calibrate the model
        self._train_calibrate(X_train, X_calib)


    def _train_calibrate(self, X_train, X_calib):
        # Fit the black-box one-class classification model on the training data
        if self.verbose:
            print("Fitting the black-box model on {:d} data points... ".format(X_train.shape[0]), end="")
            sys.stdout.flush()
        self.bbox.fit(X_train)
        if self.verbose:
            print("done.")
            sys.stdout.flush()

        # Evaluate conformity scores on the hold-out data
        if self.verbose:
            print("Calculating conformity scores for {:d} hold-out data points... ".format(X_calib.shape[0]), end="")
            sys.stdout.flush()
        self.scores_cal = self.bbox.score_samples(X_calib)
        if self.verbose:
            print("done.")
            sys.stdout.flush()


    def compute_pvalues(self, X_test):
        n_test = X_test.shape[0]
        # Compute conformity scores for all test points
        if self.verbose:
            print("Calculating conformity scores for {:d} test points... ".format(n_test), end="")
            sys.stdout.flush()
        scores_test = self.bbox.score_samples(X_test)
        if self.verbose:
            print("done.")
            sys.stdout.flush()

        # Compute conformal p-values
        n_cal = len(self.scores_cal)
        scores_mat = np.tile(self.scores_cal, (len(scores_test),1))
        tmp = np.sum(scores_mat <= scores_test.reshape(len(scores_test),1), 1)
        pvals = (1.0+tmp)/(1.0+n_cal)
        return pvals


class BinaryConformal:
    def __init__(self, X_in, X_out, bbox, calib_size=0.5, random_state=2022, verbose=True):
        self.bbox = copy.deepcopy(bbox)
        self.verbose = verbose

        # Split data into training and calibration subsets
        X_in_train, X_in_calib = train_test_split(X_in, test_size=calib_size, random_state=random_state)

        # Train and calibrate the model
        self._train_calibrate(X_in_train, X_out, X_in_calib)


    def _train_calibrate(self, X_in_train, X_out_train, X_in_calib):
        # Fit the black-box one-class classification model on the training data
        n_in_train = X_in_train.shape[0]
        n_out_train = X_out_train.shape[0]
        if self.verbose:
            print("Fitting the black-box model on {:d} inliers and {:d} outliers... ".format(n_in_train, n_out_train), end="")
            sys.stdout.flush()
        X_train = np.concatenate([X_in_train, X_out_train],0)
        Y_train = np.concatenate([[0]*n_in_train, [1]*n_out_train])
        self.bbox.fit(X_train, Y_train)
        if self.verbose:
            print("done.")
            sys.stdout.flush()

        # Evaluate conformity scores on the hold-out inliers
        if self.verbose:
            print("Calculating conformity scores for {:d} hold-out inliers... ".format(X_in_calib.shape[0]), end="")
            sys.stdout.flush()
        self.scores_cal = self.bbox.predict_proba(X_in_calib)[:,0]
        if self.verbose:
            print("done.")
            sys.stdout.flush()


    def compute_pvalues(self, X_test):
        n_test = X_test.shape[0]
        # Compute conformity scores for all test points
        if self.verbose:
            print("Calculating conformity scores for {:d} test points... ".format(n_test), end="")
            sys.stdout.flush()
        scores_test = self.bbox.predict_proba(X_test)[:,0]
        if self.verbose:
            print("done.")
            sys.stdout.flush()

        # Compute conformal p-values
        n_cal = len(self.scores_cal)
        scores_mat = np.tile(self.scores_cal, (len(scores_test),1))
        tmp = np.sum(scores_mat <= scores_test.reshape(len(scores_test),1), 1)
        pvals = (1.0+tmp)/(1.0+n_cal)
        return pvals
