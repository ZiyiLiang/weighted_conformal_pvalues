import numpy as np
import copy
import sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.stats import ranksums
import warnings

from methods_util import conformalize_scores

class IntegrativeConformal:
    '''Class used for computing the integrative conformal p-values'''
    def __init__(self, X_in, X_out, bboxes_one=None, bboxes_one_out=None, bboxes_two=None,
                 calib_size=0.5, random_state=2022, ratio=True, tuning=True, verbose=True, progress=True):
        '''Pre compute the conformity scores using specified occ/bc for the calibration points
        
        Parameters:
        -----------
        X_in:           array_like
                        inlier data
        X_out:          array_like
                        outlier data
        bboxes_one:     list
                        list of one class black boxes
        bboxes_one_out: list
                        list of one class black boxes for outlier data
        bboxes_two:     list
                        list of binary black boxes
        calib_size:     float
                        proportion of data points used for calibration, default value is 0.5
        random_state:   int
                        ensure replicability, default value is 2022
        ratio:          bool
                        If True, use weighted p-values, default value is True
        tuning:         bool
                        If True, automatically tune the conformity scores so that the outliers have smaller scores,
                        default value is True
        verbose:        bool
                        If True, print messages when training black boxes, default is True
        progress:       bool
                        If True, display progress bar, default is True
        '''
        self.tuning = tuning
        self.verbose = verbose
        self.progress = progress
        self.ratio = ratio

        if (bboxes_one is None) and (bboxes_two is None):
            print("Error: must provide at least one algorithm!")
            exit(0)
        if bboxes_one is None:
            bboxes_one = []
        if bboxes_two is None:
            bboxes_two = []

        # Split data into training and calibration subsets
        X_in_train, X_in_calib = train_test_split(X_in, test_size=calib_size, random_state=random_state)
        try:
            X_out_train, X_out_calib = train_test_split(X_out, test_size=calib_size, random_state=random_state)
        except:
            X_out_train = np.zeros((1,X_in_train.shape[1]))
            X_out_calib = np.zeros((1,X_in_train.shape[1]))

        n_in_train = X_in_train.shape[0]
        n_out_train = X_out_train.shape[0]
        X_train = np.concatenate([X_in_train, X_out_train],0)
        Y_train = np.concatenate([[0]*n_in_train, [1]*n_out_train])

        # Train all the one-class models
        self.bboxes_one_in = copy.deepcopy(bboxes_one)
        if bboxes_one_out is None:
            self.bboxes_one_out = copy.deepcopy(bboxes_one)
        else:
            self.bboxes_one_out = copy.deepcopy(bboxes_one_out)
        self.num_boxes_one = len(self.bboxes_one_in)
        self.num_boxes_one_out = len(self.bboxes_one_out)

        for b in range(self.num_boxes_one):
            self._train_one(self.bboxes_one_in[b], X_in_train)
        for b in range(self.num_boxes_one_out):
            self._train_one(self.bboxes_one_out[b], X_out_train)

        # Train all the two-class models
        self.num_boxes_two = len(bboxes_two)
        self.bboxes_two = copy.deepcopy(bboxes_two)
        for b in range(self.num_boxes_two):
            self._train_two(self.bboxes_two[b], X_train, Y_train)

        # Pre-compute conformity scores using one-class models
        self.scores_in_calib_one = np.zeros((self.num_boxes_one,X_in_calib.shape[0]))
        self.scores_outin_calib_one = np.zeros((self.num_boxes_one_out,X_in_calib.shape[0]))
        self.scores_out_calib_one = np.zeros((self.num_boxes_one_out,X_out_calib.shape[0]))
        self.scores_inout_calib_one = np.zeros((self.num_boxes_one,X_out_calib.shape[0]))

        for b in range(self.num_boxes_one):
            # Scores for inlier calibration data using inlier one-class model
            try:
                self.scores_in_calib_one[b] = self.bboxes_one_in[b].score_samples(X_in_calib)
            except:
                self.scores_in_calib_one[b] = np.ones((X_in_calib.shape[0],))
            self.scores_in_calib_one[b] += np.random.normal(loc=0, scale=1e-6, size=self.scores_in_calib_one[b].shape)
            # Scores for outlier calibration data using inlier model
            try:
                self.scores_inout_calib_one[b] = self.bboxes_one_in[b].score_samples(X_out_calib)
            except:
                self.scores_inout_calib_one[b] = np.ones((X_out_calib.shape[0],))
            self.scores_inout_calib_one[b] += np.random.normal(loc=0, scale=1e-6, size=self.scores_inout_calib_one[b].shape)

        for b in range(self.num_boxes_one_out):
            # Scores for outlier calibration data using outlier model
            try:
                self.scores_out_calib_one[b] = self.bboxes_one_out[b].score_samples(X_out_calib)
            except:
                self.scores_out_calib_one[b] = np.ones((X_out_calib.shape[0],))
            self.scores_out_calib_one[b] += np.random.normal(loc=0, scale=1e-6, size=self.scores_out_calib_one[b].shape)
            # Scores for inlier calibration data using outlier model
            try:
                self.scores_outin_calib_one[b] = self.bboxes_one_out[b].score_samples(X_in_calib)
            except:
                self.scores_outin_calib_one[b] = np.ones((X_in_calib.shape[0],))
            self.scores_outin_calib_one[b] += np.random.normal(loc=0, scale=1e-6, size=self.scores_outin_calib_one[b].shape)

        # Pre-compute conformity scores using two-class models
        self.scores_in_calib_two = np.zeros((self.num_boxes_two,X_in_calib.shape[0]))
        self.scores_out_calib_two = np.zeros((self.num_boxes_two,X_out_calib.shape[0]))
        for b in range(self.num_boxes_two):
            # Scores for inlier calibration data using inlier one-class model
            try:
                self.scores_in_calib_two[b] = self.bboxes_two[b].predict_proba(X_in_calib)[:,0]
            except:
                self.scores_in_calib_two[b] = np.ones((X_in_calib.shape[0],))
            self.scores_in_calib_two[b] += np.random.normal(loc=0, scale=1e-6, size=self.scores_in_calib_two[b].shape)
            try:
                self.scores_out_calib_two[b] = self.bboxes_two[b].predict_proba(X_out_calib)[:,0]
            except:
                self.scores_out_calib_two[b] = np.ones((X_out_calib.shape[0],))
            self.scores_out_calib_two[b] += np.random.normal(loc=0, scale=1e-6, size=self.scores_out_calib_two[b].shape)

    def _train_one(self, bbox, X_train):
        # Fit the black-box one-class classification model on the training data
        if self.verbose:
            print("Fitting a one-class classification model on {:d} data points... ".format(X_train.shape[0]), end="")
            sys.stdout.flush()
        try:
            bbox.fit(X_train)
        except:
            print("Warning: cannot train one-class classifier!")
        if self.verbose:
            print("done.")
            sys.stdout.flush()

    def _train_two(self, bbox, X_train, Y_train):
        # Fit the black-box two-class classification model on the training data
        if self.verbose:
            print("Fitting a two-class classification model on {:d} data points... ".format(X_train.shape[0]), end="")
            sys.stdout.flush()
        try:
            bbox.fit(X_train, Y_train)
        except:
            print("Warning: cannot train binary classifier!")
        if self.verbose:
            print("done.")
            sys.stdout.flush()

    def _calibrate_in(self, score_test):
        '''Compute the standard conformal p-values calibrated by the inliers.'''

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
                # Determine the seperation of inlier scores and outlier scores
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
        pvals = conformalize_scores(scores_star, scores_star, offset=0)  # Offset set to 0 since test point is included in the calibration scores.
        return pvals

    def _calibrate_out(self, score_test):
        '''Compute the standard conformal p-values calibrated by the outliers.'''

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
                # Determine the seperation of inlier scores and outlier scores
                score_contrast[b] = ranksums(scores_cal[b], scores[b])[0]

        for b in range(self.num_boxes_two):
            b_tot = b + self.num_boxes_one
            scores_cal[b_tot] = self.scores_out_calib_two[b]
            # Concatenate conformity scores for inlier calibration data and test point
            scores[b_tot] = np.append(self.scores_in_calib_two[b], score_test[b_tot])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Determine the seperation of inlier scores and outlier scores
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

    def _compute_pvalue(self, score_in_test, score_out_test):
        '''Compute the integrative conformal p-value for one test point.'''
        pvals_0 = self._calibrate_in(score_in_test)
        if self.ratio:
            pvals_1 = self._calibrate_out(score_out_test)
        else:
            pvals_1 = np.ones((len(pvals_0),))
        scores = pvals_0 / pvals_1
        # Compute final conformal p-value
        pvals = conformalize_scores(scores, scores, offset=0)
        # return the integrative conformal p-value and two pre p-vals for the test point
        return pvals[-1], pvals_0[-1], pvals_1[-1]

    def compute_pvalues(self, X_test, return_prepvals=False):
        '''Compute the integrative conformal p-values for all test points.'''
        n_test = X_test.shape[0]
        num_boxes = self.num_boxes_one + self.num_boxes_two
        num_boxes_out = self.num_boxes_one_out + self.num_boxes_two
        self.scores_in_test = np.zeros((n_test,num_boxes))
        self.scores_out_test = np.zeros((n_test,num_boxes_out))

        # Compute conformity scores for test data
        for b in range(self.num_boxes_one):
            try:
                self.scores_in_test[:,b] = self.bboxes_one_in[b].score_samples(X_test)
            except:
                self.scores_in_test[:,b] = 1
            self.scores_in_test[:,b] += np.random.normal(loc=0, scale=1e-6, size=self.scores_in_test[:,b].shape)

        for b in range(self.num_boxes_one_out):
            try:
                self.scores_out_test[:,b] = self.bboxes_one_out[b].score_samples(X_test)
            except:
                self.scores_out_test[:,b] = 1
            self.scores_out_test[:,b] += np.random.normal(loc=0, scale=1e-6, size=self.scores_out_test[:,b].shape)
        for b in range(self.num_boxes_two):
            b_tot = b + self.num_boxes_one
            try:
                self.scores_in_test[:,b_tot] = self.bboxes_two[b].predict_proba(X_test)[:,0]
            except:
                self.scores_in_test[:,b_tot] = 1
            self.scores_in_test[:,b_tot] += np.random.normal(loc=0, scale=1e-6, size=self.scores_in_test[:,b_tot].shape)
            try:
                self.scores_out_test[:,b_tot] = self.bboxes_two[b].predict_proba(X_test)[:,0]
            except:
                self.scores_out_test[:,b_tot] = 1
            self.scores_out_test[:,b_tot] += np.random.normal(loc=0, scale=1e-6, size=self.scores_out_test[:,b_tot].shape)

        if self.progress:
            iterator = tqdm(range(n_test))
        else:
            iterator = range(n_test)

        pvals = -np.ones((n_test,))
        pvals_0 = -np.ones((n_test,))
        pvals_1 = -np.ones((n_test,))
        for i in iterator:
            pvals[i], pvals_0[i], pvals_1[i] = self._compute_pvalue(self.scores_in_test[i], self.scores_out_test[i])

        if return_prepvals:
            return pvals, pvals_0, pvals_1
        else:
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

        pvals = conformalize_scores(self.scores_cal, scores_test, offset=1)
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
        try:
            self.bbox.fit(X_train, Y_train)
        except:
            print("Warning: cannot train binary classifier!")
        if self.verbose:
            print("done.")
            sys.stdout.flush()

        # Evaluate conformity scores on the hold-out inliers
        if self.verbose:
            print("Calculating conformity scores for {:d} hold-out inliers... ".format(X_in_calib.shape[0]), end="")
            sys.stdout.flush()
        try:
            self.scores_cal = self.bbox.predict_proba(X_in_calib)[:,0]
        except:
            self.scores_cal = np.ones((X_in_calib.shape[0],))
        self.scores_cal += np.random.normal(loc=0, scale=1e-6, size=self.scores_cal.shape)
        if self.verbose:
            print("done.")
            sys.stdout.flush()


    def compute_pvalues(self, X_test):
        n_test = X_test.shape[0]
        # Compute conformity scores for all test points
        if self.verbose:
            print("Calculating conformity scores for {:d} test points... ".format(n_test), end="")
            sys.stdout.flush()
        try:
            scores_test = self.bbox.predict_proba(X_test)[:,0]
        except:
            scores_test = np.ones((X_test.shape[0],))
        scores_test += np.random.normal(loc=0, scale=1e-6, size=scores_test.shape)
        if self.verbose:
            print("done.")
            sys.stdout.flush()

        # Compute conformal p-values
        pvals = conformalize_scores(self.scores_cal, scores_test, offset=1)
        return pvals
