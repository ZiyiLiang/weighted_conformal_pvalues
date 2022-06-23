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

class IntegrativeConformal:
    def __init__(self, X_in, X_out, bboxes_one=None, bboxes_one_out=None, bboxes_two=None, bboxes_two_out=None,
                 n_folds=10, random_state=2022, ratio=True, tuning=True, verbose=True, progress=True):
        assert n_folds > 1

        self.tuning = True
        self.verbose = verbose
        self.progress = progress
        self.ratio = ratio
        self.n_folds = n_folds
        self.X_in = X_in
        self.X_out = X_out

        if (bboxes_one is None) and (bboxes_two is None):
            print("Error: must provide at least one algorithm!")
            exit(0)
        if bboxes_one is None:
            bboxes_one = []
        if bboxes_two is None:
            bboxes_two = []

        # Split the inliers and placeholder test point into folds
        X_test_dummy = np.zeros((1,len(X_in[0])))
        X_intest = np.concatenate([X_in, X_test_dummy],0)
        cv = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        self.folds_in = [(train_idx, cal_idx) for train_idx, cal_idx in cv.split(X_intest)]
        self.n_folds_in = len(self.folds_in)

        # Split the outliers into folds
        cv = KFold(n_splits=n_folds, random_state=random_state+1, shuffle=True)
        self.folds_out = [(train_idx, cal_idx) for train_idx, cal_idx in cv.split(X_out)]
        self.n_folds_out = len(self.folds_out)

        # Pre-assign the test point to the last fold
        self.test_fold_id = self.n_folds-1


        # Initialize the one-class models for outliers
        if bboxes_one_out is None:
            self.bboxes_one_out = [[copy.deepcopy(box) for k in range(self.n_folds)] for box in bboxes_one]
        else:
            self.bboxes_one_out = [[copy.deepcopy(box) for k in range(self.n_folds)] for box in bboxes_one_out]
        self.num_boxes_one_out = len(self.bboxes_one_out)

        # Train the one-class models for outliers
        if self.ratio:
            for b in range(self.num_boxes_one_out):
                for k in range(self.n_folds_out):
                    train_idx = self.folds_out[k][0]
                    n_train = len(train_idx)
                    if self.verbose:
                        print("Fitting the black-box model {:d} on fold {:d}, which contains {:d} outliers... ".format(b, k, n_train), end="")
                        sys.stdout.flush()
                    try:
                        self.bboxes_one_out[b][k].fit(X_out[train_idx])
                    except:
                        print("Warning: cannot train OCC!")
                    if self.verbose:
                        print("done.")
                        sys.stdout.flush()

        # Pre-compute calibration scores for outliers using outlier models
        self.scores_out_calib_one = np.zeros((self.num_boxes_one_out,X_out.shape[0]))
        for b in range(self.num_boxes_one_out):
            for k in range(self.n_folds_out):
                cal_idx = self.folds_out[k][1]
                n_cal = len(cal_idx)
                try:
                    self.scores_out_calib_one[b,cal_idx] = self.bboxes_one_out[b][k].score_samples(X_out[cal_idx])
                except:
                    self.scores_out_calib_one[b,cal_idx] = 1
                self.scores_out_calib_one[b,cal_idx] += np.random.normal(loc=0, scale=1e-6, size=(len(cal_idx),))

        # Pre-compute calibration scores for inliers using outlier models
        self.scores_outin_calib_one = np.zeros((self.num_boxes_one_out,self.n_folds_out,X_in.shape[0]))
        for b in range(self.num_boxes_one_out):
            for k in range(self.n_folds_out):
                try:
                    self.scores_outin_calib_one[b,k] = self.bboxes_one_out[b][k].score_samples(X_in)
                except:
                    self.scores_outin_calib_one[b,k] = 1
                self.scores_outin_calib_one[b,k] += np.random.normal(loc=0, scale=1e-6, size=(X_in.shape[0],))


        # Initialize the one-class models for inliers
        self.bboxes_one_in = [[copy.deepcopy(box) for k in range(self.n_folds)] for box in bboxes_one]
        self.num_boxes_one_in = len(self.bboxes_one_in)

    def compute_pvalue_single(self, X_test, return_prepvals=False):
        def _calibrate_in():
            # Augment the inlier data with the test point
            X_intest = np.concatenate([self.X_in, X_test],0)

            # Train the one-class models for inliers
            for b in range(self.num_boxes_one_in):
                for k in range(self.n_folds_in):
                    train_idx = self.folds_in[k][0]
                    n_train = len(train_idx)
                    try:
                        self.bboxes_one_in[b][k].fit(X_intest[train_idx])
                    except:
                        print("Warning: cannot train OCC!")

            # Pre-compute calibration scores for inliers using inlier models
            scores_caltest = np.zeros((self.num_boxes_one_in,X_intest.shape[0]))
            for b in range(self.num_boxes_one_in):
                for k in range(self.n_folds_in):
                    cal_idx = self.folds_in[k][1]
                    n_cal = len(cal_idx)
                    try:
                        scores_caltest[b,cal_idx] = self.bboxes_one_in[b][k].score_samples(X_intest[cal_idx])
                    except:
                        scores_caltest[b,cal_idx] = 1
                    scores_caltest[b,cal_idx] += np.random.normal(loc=0, scale=1e-6, size=(len(cal_idx),))

            # Pre-compute calibration scores for outliers using inliers models
            self.scores_inout_calib_one = np.zeros((self.num_boxes_one_in,self.n_folds_in,self.X_out.shape[0]))
            for b in range(self.num_boxes_one_in):
                for k in range(self.n_folds_in):
                    try:
                        self.scores_inout_calib_one[b,k] = self.bboxes_one_in[b][k].score_samples(self.X_out)
                    except:
                        self.scores_inout_calib_one[b,k] = 1
                    self.scores_inout_calib_one[b,k] += np.random.normal(loc=0, scale=1e-6, size=(self.X_out.shape[0],))


            # Estimate the performance of each model
            score_contrast = np.zeros((self.num_boxes_one_in,))
            for b in range(self.num_boxes_one_in):
                # Change the direction of the scores, if necessary
                if self.tuning:
                    # Large score <-> large p-value
                    # We expect the outliers should have smaller scores
                    # Concatenate conformity scores for inlier calibration data and test point
                    median_out = np.median(self.scores_inout_calib_one[b])
                    median_caltest = np.median(scores_caltest[b])
                    if median_caltest < median_out:
                        scores_caltest[b] = - scores_caltest[b]
                        self.scores_inout_calib_one[b] = - self.scores_inout_calib_one[b]
                # Compute score contrast
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for k in range(self.n_folds_in):
                        cal_idx = self.folds_in[k][1]
                        scores_out = self.scores_inout_calib_one[b,k]
                        score_contrast[b] += ranksums(scores_caltest[b][cal_idx], scores_out)[0]

            # Pick the best OCC model
            b_star = np.argmax(score_contrast)

            # Compute the preliminary p-values using the best inlier model
            pvals = conformalize_scores(scores_caltest[b_star], scores_caltest[b_star], offset=0)

            # TODO: make sure this is correct
            return pvals

        def _calibrate_out():
            # Compute scores for test point using outlier models
            scores_test = np.zeros((self.num_boxes_one_out,self.n_folds_out))
            for b in range(self.num_boxes_one_out):
                for k in range(len(self.folds_out)):
                    try:
                        scores_test[b,k] =  self.bboxes_one_out[b][k].score_samples(X_test)
                    except:
                        scores_test[b,k] = 1
                    scores_test[b,k] += np.random.normal(loc=0, scale=1e-6, size=(1,))

            # Estimate the performance of each model
            score_contrast = np.zeros((self.num_boxes_one_out,))
            for b in range(self.num_boxes_one_out):
                scores_intest = np.concatenate([self.scores_outin_calib_one[b],scores_test[b,None].T],1)
                # Change the direction of the scores, if necessary
                if self.tuning:
                    median_cal = 0
                    median_intest = 0
                    for k in range(len(self.folds_out)):
                        fold = self.folds_out[k][1]
                        median_cal += np.median(self.scores_out_calib_one[b][fold])
                        median_intest += np.median(scores_intest[k])
                    # Large score <-> large p-value
                    # We expect the inliers should have smaller scores
                    if median_intest > median_cal:
                        self.scores_out_calib_one[b] = -self.scores_out_calib_one[b]
                        scores_test[b] = - scores_test[b]
                        self.scores_outin_calib_one[b] = -self.scores_outin_calib_one[b]
                        scores_intest = - scores_intest

                # Compute score contrast
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for k in range(self.n_folds_out):
                        fold = self.folds_out[k]
                        cal_idx = fold[1]
                        scores_cal = self.scores_out_calib_one[b][cal_idx]
                        score_contrast[b] += ranksums(scores_cal, scores_intest[k])[0]

            # Pick the best OCC model
            b_star = np.argmax(score_contrast)              

            # Compute the preliminary p-values using the best outlier model
            scores_cal = self.scores_out_calib_one[b_star]
            scores_intest = np.concatenate([self.scores_outin_calib_one[b_star],scores_test[b_star,None].T],1)
            pvals = -np.ones((self.X_in.shape[0]+1,))
            for i in range(len(pvals)):
                scores_test_tmp = -np.ones((self.X_out.shape[0],))
                for k in range(self.n_folds_out):
                    fold = self.folds_out[k][1]
                    scores_test_tmp[fold] = scores_intest[k][i]
                            
                    pvals[i] = (1.0+np.sum(scores_cal <= scores_test_tmp)) / (1.0+len(scores_cal))
            
            return pvals


        pvals_0 = _calibrate_in()
        if self.ratio:
            pvals_1 = _calibrate_out()
        else:
            pvals_1 = np.ones(pvals_0.shape)

        # Compute final conformal p-value
        scores = pvals_0 / pvals_1
        pvals = conformalize_scores(scores, scores, offset=0)

        return pvals[-1], pvals_0[-1], pvals_1[-1]


    def compute_pvalues(self, X_test, return_prepvals=False):
        n_test = X_test.shape[0]

        pvals = -np.ones((n_test,))
        pvals_0 = -np.ones((n_test,))
        pvals_1 = -np.ones((n_test,))

        if self.progress:
            iterator = tqdm(range(n_test))
        else:
            iterator = range(n_test)

        for i in iterator:
            pvals[i], pvals_0[i], pvals_1[i] = self.compute_pvalue_single(X_test[i,None])

        if return_prepvals:
            return pvals, pvals_0, pvals_1
        else:
            return pvals


        # num_boxes_in = self.num_boxes_one + self.num_boxes_two
        # num_boxes_out = self.num_boxes_one_out + self.num_boxes_two

        # # Initialize containers for
        # scores_in_test = np.zeros((n_test,num_boxes))
        # scores_out_test = np.zeros((n_test,num_boxes_out))

        # # Compute conformity scores for test data
        # for b in range(self.num_boxes_one):
        #     try:
        #         scores_in_test[:,b] = self.bboxes_one_in[b].score_samples(X_test)
        #     except:
        #         scores_in_test[:,b] = 1
        #     scores_in_test[:,b] += np.random.normal(loc=0, scale=1e-6, size=scores_in_test[:,b].shape)

        # for b in range(self.num_boxes_one_out):
        #     try:
        #         scores_out_test[:,b] = self.bboxes_one_out[b].score_samples(X_test)
        #     except:
        #         scores_out_test[:,b] = 1
        #     scores_out_test[:,b] += np.random.normal(loc=0, scale=1e-6, size=scores_out_test[:,b].shape)
        # for b in range(self.num_boxes_two):
        #     b_tot = b + self.num_boxes_one
        #     try:
        #         scores_in_test[:,b_tot] = self.bboxes_two[b].predict_proba(X_test)[:,0]
        #     except:
        #         scores_in_test[:,b_tot] = 1
        #     scores_in_test[:,b_tot] += np.random.normal(loc=0, scale=1e-6, size=scores_in_test[:,b_tot].shape)
        #     try:
        #         scores_out_test[:,b_tot] = self.bboxes_two[b].predict_proba(X_test)[:,0]
        #     except:
        #         scores_out_test[:,b_tot] = 1
        #     scores_out_test[:,b_tot] += np.random.normal(loc=0, scale=1e-6, size=scores_out_test[:,b_tot].shape)

        # def compute_pvalue(score_in_test, score_out_test):
        #     pvals_0 = self._calibrate_in(score_in_test)
        #     if self.ratio:
        #         pvals_1 = self._calibrate_out(score_out_test)
        #     else:
        #         pvals_1 = np.ones((len(pvals_0),))
        #     scores = pvals_0 / pvals_1
        #     # Compute final conformal p-value
        #     pvals = conformalize_scores(scores, scores, offset=0)
        #     return pvals[-1], pvals_0[-1], pvals_1[-1]



    def _calibrate_in(self, score_test):
        n_calib_in = self.scores_in_calib_one.shape[1]
        n_calib_out = self.scores_out_calib_one.shape[1]
        num_boxes = self.num_boxes_one + self.num_boxes_two
        scores = np.zeros((num_boxes,n_calib_in+1))
        scores_out = np.zeros((num_boxes,n_calib_out))
        score_contrast = np.zeros((num_boxes,))

        for b in range(self.num_boxes_one):
            scores_out[b] = self.scores_inout_calib_one[b]

            # Concatenate conformity scores for inlier calibration data and test point
            scores[b] = np.append(self.scores_in_calib_one[b], score_test[b])

            if self.tuning:
                # Large score <-> large p-value
                # We expect the outliers should have smaller scores
                median_out = np.median(scores_out[b])
                median_in = np.median(scores[b])
                if median_in < median_out:
                    scores[b] = -scores[b]
                    scores_out[b] = -scores_out[b]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score_contrast[b] = ranksums(scores[b], scores_out[b])[0]

        for b in range(self.num_boxes_two):
            b_tot = b + self.num_boxes_one
            scores_out[b_tot] = self.scores_out_calib_two[b]
            # Concatenate conformity scores for inlier calibration data and test point
            scores[b_tot] = np.append(self.scores_in_calib_two[b], score_test[b_tot])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score_contrast_new = ranksums(scores[b_tot], scores_out[b_tot])[0]
            if np.isnan(score_contrast_new):
                score_contrast_new = -np.inf
            score_contrast[b_tot] = score_contrast_new

        # Pick the best model
        b_star = np.argmax(score_contrast)
        scores_star = scores[b_star]

        # Compute conformal p-values using the scores from the best model
        pvals = conformalize_scores(scores_star, scores_star, offset=0)
        return pvals

    def _calibrate_out(self, score_test):
        n_calib_in = self.scores_in_calib_one.shape[1]
        n_calib_out = self.scores_out_calib_one.shape[1]
        num_boxes = self.num_boxes_one_out + self.num_boxes_two
        scores = np.zeros((num_boxes,n_calib_in+1))
        scores_cal = np.zeros((num_boxes,n_calib_out))
        score_contrast = np.zeros((num_boxes,))

        for b in range(self.num_boxes_one_out):
            scores_cal[b] = self.scores_out_calib_one[b]

            # Concatenate conformity scores based on outlier model for inlier calibration data and test point
            scores[b] = np.append(self.scores_outin_calib_one[b], score_test[b])

            if self.tuning:
                # Large score <-> large p-value
                # We expect the inliers should have smaller scores
                median_out = np.median(scores_cal[b])
                median_in = np.median(scores[b])
                if median_in > median_out:
                    scores[b] = -scores[b]
                    scores_cal[b] = -scores_cal[b]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score_contrast[b] = ranksums(scores_cal[b], scores[b])[0]

        for b in range(self.num_boxes_two):
            b_tot = b + self.num_boxes_one
            scores_cal[b_tot] = self.scores_out_calib_two[b]
            # Concatenate conformity scores for inlier calibration data and test point
            scores[b_tot] = np.append(self.scores_in_calib_two[b], score_test[b_tot])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score_contrast_new = ranksums(scores_cal[b_tot], scores[b_tot])[0]
            if np.isnan(score_contrast_new):
                score_contrast_new = -np.inf
            score_contrast[b_tot] = score_contrast_new

        # Pick the best model
        b_star = np.argmax(score_contrast)
        scores_star = scores[b_star]
        scores_cal_star = scores_cal[b_star]

        # Compute conformal p-values
        pvals = conformalize_scores(scores_cal_star, scores_star, offset=1)
        return pvals


class BinaryConformal:
    def __init__(self, X_in, X_out, bbox, n_folds=10, random_state=2022, verbose=True):
        self.verbose = verbose
        self.n_folds = n_folds
        self.bboxes = [copy.deepcopy(bbox) for k in range(self.n_folds)]

        # Split the inlier data into training and calibration subsets
        cv = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        self.folds = [(train_idx, cal_idx) for train_idx, cal_idx in cv.split(X_in)]

        # Train and calibrate the model
        self._train_calibrate(X_in, X_out)


    def _train_calibrate(self, X_in, X_out):
        self.n_cal = X_in.shape[0]
        # Train the binary classification models
        self.models = [None] * self.n_folds
        for k in range(self.n_folds):
            train_idx = self.folds[k][0]
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

        # Evaluate conformity scores on the hold-out inliers
        self.scores_cal = -np.ones((self.n_cal,))
        for k in range(self.n_folds):
            cal_idx = self.folds[k][1]
            try:
                self.scores_cal[cal_idx] = self.bboxes[k].predict_proba(X_in[cal_idx])[:,0]
            except:
                self.scores_cal[cal_idx] = 1
            self.scores_cal[cal_idx] += np.random.normal(loc=0, scale=1e-6, size=(len(cal_idx),))

    def compute_pvalues(self, X_test):
        n_test = X_test.shape[0]
        scores_test = -np.ones((n_test,self.n_cal))
        for k in range(self.n_folds):
            cal_idx = self.folds[k][1]
            try:
                scores_tmp = self.bboxes[k].predict_proba(X_test)[:,0]
            except:
                scores_tmp = 1
            scores_tmp += np.random.normal(loc=0, scale=1e-6, size=(len(scores_tmp),))
            scores_test[:,cal_idx] = np.tile(scores_tmp, (len(cal_idx),1)).T

        scores_cal_mat = np.tile(self.scores_cal, (n_test,1))
        pvals = (1.0+np.sum(scores_cal_mat <= scores_test,1))/(self.n_cal+1.0)
        return pvals

class OneClassConformal:
    def __init__(self, X_in, bbox, n_folds=10, random_state=2022, verbose=True):
        self.verbose = verbose
        self.n_folds = n_folds
        self.bboxes = [copy.deepcopy(bbox) for k in range(self.n_folds)]

        # Split data into training and calibration subsets
        cv = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        self.folds = [(train_idx, cal_idx) for train_idx, cal_idx in cv.split(X_in)]

        # Train and calibrate the model
        self._train_calibrate(X_in)


    def _train_calibrate(self, X_in):
        self.n_cal = X_in.shape[0]
        # Train the binary classification models
        self.models = [None] * self.n_folds
        for k in range(self.n_folds):
            train_idx = self.folds[k][0]
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

        # Evaluate conformity scores on the hold-out inliers
        self.scores_cal = -np.ones((self.n_cal,))
        for k in range(self.n_folds):
            cal_idx = self.folds[k][1]
            try:
                self.scores_cal[cal_idx] = self.bboxes[k].score_samples(X_in[cal_idx])
            except:
                self.scores_cal[cal_idx] = 1
            self.scores_cal[cal_idx] += np.random.normal(loc=0, scale=1e-6, size=(len(cal_idx),))

    def compute_pvalues(self, X_test):
        n_test = X_test.shape[0]
        scores_test = -np.ones((n_test,self.n_cal))
        for k in range(self.n_folds):
            cal_idx = self.folds[k][1]
            try:
                scores_tmp = self.bboxes[k].score_samples(X_test)
            except:
                scores_tmp = 1
            scores_tmp += np.random.normal(loc=0, scale=1e-6, size=(len(scores_tmp),))
            scores_test[:,cal_idx] = np.tile(scores_tmp, (len(cal_idx),1)).T

        scores_cal_mat = np.tile(self.scores_cal, (n_test,1))
        pvals = (1.0+np.sum(scores_cal_mat <= scores_test,1))/(self.n_cal+1.0)
        return pvals
