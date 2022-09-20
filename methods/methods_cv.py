import numpy as np
import copy
import sys
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy.stats import ranksums
import warnings

from methods_util import conformalize_scores

class IntegrativeConformal:
    '''Class used for computing the integrative conformal p-values with transductive cross-validation+'''
    def __init__(self, X_in, X_out, bboxes_one=None, bboxes_one_out=None, bboxes_two=None,
                 n_folds=5, random_state=2022, ratio=True, tuning=True, verbose=True, progress=True):
        '''Initialize OCC/BC models 
        
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
        n_folds:        int
                        number of cross validation folds
        random_state:   int
                        ensure replicability, default value is 2022
        ratio:          bool
                        If True, use weighted p-values, else use standard conformal p-values, default value is True
        tuning:         bool
                        If True, automatically tune the conformity scores so that the outliers have smaller scores,
                        default value is True
        verbose:        bool
                        If True, print messages when training black boxes, default is True
        progress:       bool
                        If True, display progress bar, default is True
        '''
        assert n_folds > 1

        self.tuning = tuning
        self.verbose = verbose
        self.progress = progress
        self.ratio = ratio
        self.X_in = X_in
        self.X_out = X_out

        if (bboxes_one is None) and (bboxes_two is None):
            print("Error: must provide at least one algorithm!")
            exit(0)
        if bboxes_one is None:
            bboxes_one = []
        if bboxes_two is None:
            bboxes_two = []

        n_folds_in = int(np.maximum(2, np.minimum(n_folds, X_in.shape[0])))
        n_folds_out = int(np.maximum(2, np.minimum(n_folds, X_out.shape[0])))
        self.n_folds = np.minimum(n_folds_in, n_folds_out)

        # Split the inliers and placeholder test point into folds
        X_test_dummy = np.zeros((1,len(X_in[0])))
        X_intest = np.concatenate([X_in, X_test_dummy],0)
        cv = KFold(n_splits=self.n_folds, random_state=random_state, shuffle=True)
        self.folds_in = [(train_idx, cal_idx) for train_idx, cal_idx in cv.split(X_intest)]

        # Split the outliers into folds
        cv = KFold(n_splits=self.n_folds, random_state=random_state+1, shuffle=True)
        self.folds_out = [(train_idx, cal_idx) for train_idx, cal_idx in cv.split(X_out)]

        # Initialize the one-class models for outliers
        if bboxes_one_out is None:
            self.bboxes_one_out = [[copy.deepcopy(box) for k in range(self.n_folds)] for box in bboxes_one]
        else:
            self.bboxes_one_out = [[copy.deepcopy(box) for k in range(self.n_folds)] for box in bboxes_one_out]
        self.num_boxes_one_out = len(self.bboxes_one_out)

        # Train the one-class models for outliers
        if self.ratio:
            for b in range(self.num_boxes_one_out):
                for k in range(self.n_folds):
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
            for k in range(self.n_folds):
                cal_idx = self.folds_out[k][1]
                try:
                    self.scores_out_calib_one[b,cal_idx] = self.bboxes_one_out[b][k].score_samples(X_out[cal_idx])
                except:
                    self.scores_out_calib_one[b,cal_idx] = 1
                self.scores_out_calib_one[b,cal_idx] += np.random.normal(loc=0, scale=1e-6, size=(len(cal_idx),))

        # Pre-compute calibration scores for inliers using outlier models
        self.scores_outin_calib_one = np.zeros((self.num_boxes_one_out,self.n_folds,X_in.shape[0]))
        for b in range(self.num_boxes_one_out):
            for k in range(self.n_folds):
                try:
                    self.scores_outin_calib_one[b,k] = self.bboxes_one_out[b][k].score_samples(X_in)
                except:
                    self.scores_outin_calib_one[b,k] = 1
                self.scores_outin_calib_one[b,k] += np.random.normal(loc=0, scale=1e-6, size=(X_in.shape[0],))


        # Initialize the one-class models for inliers (these will be trained once we have a test point)
        self.bboxes_one_in = [[copy.deepcopy(box) for k in range(self.n_folds)] for box in bboxes_one]
        self.num_boxes_one_in = len(self.bboxes_one_in)

        # Initialize the binary classification models (these will be trained once we have a test point)
        self.bboxes_two = [[copy.deepcopy(box) for k in range(self.n_folds)] for box in bboxes_two]
        self.num_boxes_two = len(self.bboxes_two)
        

    def compute_pvalue_single(self, X_test):
        '''Compute the integrative conformal p-value for one test point.'''
        def _calibrate_in():
            # Augment the inlier data with the test point
            X_intest = np.concatenate([self.X_in, X_test],0)

            # Train the one-class models for inliers
            for b in range(self.num_boxes_one_in):
                for k in range(self.n_folds):
                    train_idx = self.folds_in[k][0]
                    try:
                        self.bboxes_one_in[b][k].fit(X_intest[train_idx])
                    except:
                        print("Warning: cannot train OCC!")

            # Train the binary classification models
            for b in range(self.num_boxes_two):
                for k in range(self.n_folds):
                    train_idx_in = self.folds_in[k][0]
                    train_idx_out = self.folds_out[k][0]
                    X_train = np.concatenate([X_intest[train_idx_in], self.X_out[train_idx_out]],0)
                    Y_train = np.concatenate([[0]*len(train_idx_in), [1]*len(train_idx_out)])
                    try:
                        self.bboxes_two[b][k].fit(X_train, Y_train)
                    except:
                        print("Warning: cannot train OCC!")
            
            # Initialize calibration scores for inliers using all models
            num_boxes = self.num_boxes_one_in + self.num_boxes_two
            scores_caltest = np.zeros((num_boxes,X_intest.shape[0]))

            # Compute calibration scores for inliers using one-class inlier models in each fold
            for b in range(self.num_boxes_one_in):
                for k in range(self.n_folds):
                    cal_idx = self.folds_in[k][1]
                    try:
                        scores_caltest[b,cal_idx] = self.bboxes_one_in[b][k].score_samples(X_intest[cal_idx])
                    except:
                        scores_caltest[b,cal_idx] = 1
                    scores_caltest[b,cal_idx] += np.random.normal(loc=0, scale=1e-6, size=(len(cal_idx),))

            # Compute calibration scores for inliers using binary models in each fold
            for b2 in range(self.num_boxes_two):
                b = b2 + self.num_boxes_one_in
                for k in range(self.n_folds):
                    cal_idx = self.folds_in[k][1]
                    try:
                        scores_caltest[b,cal_idx] = self.bboxes_two[b2][k].predict_proba(X_intest[cal_idx])[:,0]
                    except:
                        scores_caltest[b,cal_idx] = 1
                    scores_caltest[b,cal_idx] += np.random.normal(loc=0, scale=1e-6, size=(len(cal_idx),))

            # Initialize calibration scores for outliers using all inlier models, note that these scores are for model selection
            scores_out = np.zeros((num_boxes,self.n_folds,self.X_out.shape[0]))

            # Compute calibration scores for outliers using one-class inliers models in each fold
            for b in range(self.num_boxes_one_in):
                for k in range(self.n_folds):
                    try:
                        scores_out[b,k] = self.bboxes_one_in[b][k].score_samples(self.X_out)
                    except:
                        scores_out[b,k] = 1
                    scores_out[b,k] += np.random.normal(loc=0, scale=1e-6, size=(self.X_out.shape[0],))

            # Compute calibration scores for outliers using binary models in each fold
            for b2 in range(self.num_boxes_two):
                b = b2 + self.num_boxes_one_in
                for k in range(self.n_folds):
                    train_idx_out = self.folds_out[k][0]
                    cal_idx_out = self.folds_out[k][1]
                    try:
                        scores_out[b,k,cal_idx_out] = self.bboxes_two[b2][k].predict_proba(self.X_out[cal_idx_out])[:,0]
                        scores_out[b,k,train_idx_out] = np.median(scores_caltest[b])
                    except:
                        scores_out[b,k] = 1
                    scores_out[b,k] += np.random.normal(loc=0, scale=1e-6, size=(self.X_out.shape[0],))

            # Estimate the performance of each model
            score_contrast = np.zeros((num_boxes,))
            for b in range(num_boxes):
                # Change the direction of the scores, if necessary (do not do this for binary classifiers)
                if (self.tuning) and (b<self.num_boxes_one_out):
                    # Large score <-> large p-value
                    # We expect the outliers should have smaller scores
                    # Concatenate conformity scores for inlier calibration data and test point
                    median_out = np.median(scores_out[b])
                    median_caltest = np.median(scores_caltest[b])
                    if median_caltest < median_out:
                        scores_caltest[b] = - scores_caltest[b]
                        scores_out[b] = - scores_out[b]
                # Compute score contrast for each fold
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for k in range(self.n_folds):
                        cal_idx = self.folds_in[k][1]
                        score_contrast[b] += ranksums(scores_caltest[b][cal_idx], scores_out[b,k])[0]

            # Pick the best model
            b_star = np.argmax(score_contrast)
            if self.verbose:
                print("Best model for inliers: {:d}".format(b_star))

            # Compute the preliminary p-values using the best inlier model
            pvals = conformalize_scores(scores_caltest[b_star], scores_caltest[b_star], offset=0)

            return pvals

        def _calibrate_out():
            # Compute scores for test point using outlier models
            scores_test = np.zeros((self.num_boxes_one_out,self.n_folds))
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
                # Change the direction of the scores, if necessary, recall that scores are already
                # calculated in the class initialization stage
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
                    for k in range(self.n_folds):
                        fold = self.folds_out[k]
                        cal_idx = fold[1]
                        scores_cal = self.scores_out_calib_one[b][cal_idx]
                        score_contrast[b] += ranksums(scores_cal, scores_intest[k])[0]

            # Pick the best model
            b_star = np.argmax(score_contrast) 
            if self.verbose:             
                print("Best model for outliers: {:d}".format(b_star))

            # Choose the scores from the best outlier model
            scores_cal = self.scores_out_calib_one[b_star]
            scores_intest = np.concatenate([self.scores_outin_calib_one[b_star],scores_test[b_star,None].T],1)
            pvals = -np.ones((self.X_in.shape[0]+1,))
            # Compute preliminairy p-values for inliers and test point one at a time
            for i in range(len(pvals)):
                scores_test_tmp = -np.ones((self.X_out.shape[0],))
                # Compare infold scores
                for k in range(self.n_folds):
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
        '''Compute the integrative conformal p-value for all test points.'''
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



class BinaryConformal:
    '''Class used for computing standard conformal p-values with binary classifiers and cross validation+'''
    def __init__(self, X_in, X_out, bbox, n_folds=10, random_state=2022, verbose=True):
        '''Split the folds, train and compute the calibration scores for each fold
        
        Parameters:
        -----------
        X_in:           array_like
                        inlier data
        X_out:          array_like
                        outlier data
        bbox:           class object
                        a binary classifier
        n_folds:        int
                        number of cross validation folds, default value is 10
        random_state:   int
                        ensure replicability, default value is 2022
        verbose:        bool
                        If True, print messages when training black boxes, default is True
        '''
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
    '''Class used for computing standard conformal p-values with one class classifiers and cross validation+'''
    def __init__(self, X_in, bbox, n_folds=10, random_state=2022, verbose=True):
        '''Split the folds, train and compute the calibration scores for each fold
        
        Parameters:
        -----------
        X_in:           array_like
                        inlier data
        bbox:           class object
                        a one class classifier
        n_folds:        int
                        number of cross validation folds, default value is 10
        random_state:   int
                        ensure replicability, default value is 2022
        verbose:        bool
                        If True, print messages when training black boxes, default is True
        '''
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
