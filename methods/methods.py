import numpy as np
import statistics as stats
import copy
import sys
from sklearn.model_selection import train_test_split

sys.path.append('../third_party')

class OneClassConformal:
    def __init__(self, X_inliers, bbox, calib_size=0.5, random_state=2020, verbose=True):
        self.bbox = copy.deepcopy(bbox)
        self.verbose = verbose

        # Split data into training and calibration subsets
        self.X_train, self.X_calib = train_test_split(X_inliers, test_size=calib_size, random_state=random_state)

        # Train and calibrate the model
        self._train_calibrate()


    def _train_calibrate(self):
        # Fit the black-box one-class classification model on the training data
        if self.verbose:
            print("Fitting the black-box model on {:d} data points... ".format(self.X_train.shape[0]), end="")
            sys.stdout.flush()
        self.bbox.fit(self.X_train)
        if self.verbose:
            print("done.")
            sys.stdout.flush()

        # Evaluate conformity scores on the hold-out data
        if self.verbose:
            print("Calculating conformity scores for {:d} data points... ".format(self.X_calib.shape[0]), end="")
            sys.stdout.flush()
        self.scores_cal = self.bbox.score_samples(self.X_calib)
        if self.verbose:
            print("done.")
            sys.stdout.flush()
        self.n_cal = len(self.scores_cal)

                    
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
        scores_mat = np.tile(self.scores_cal, (len(scores_test),1))
        tmp = np.sum(scores_mat <= scores_test.reshape(len(scores_test),1), 1)
        pvals = (1.0+tmp)/(1.0+self.n_cal)
        return pvals
