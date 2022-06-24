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
from statsmodels.stats.multitest import multipletests

from methods_split import IntegrativeConformal

class IntegrativeConformalFDR:
    def __init__(self, ic):
        self.ic = ic

    def _estimate_R_tilde_single(self, i, j, alpha):
        n_test = self.ic.scores_in_test.shape[0]
        n_cal = self.ic.scores_in_calib_one.shape[1]
        assert( (j>=0) and (j<n_cal))
        # Extract the conformity scores for the i-th test point
        scores_in_test = self.ic.scores_in_test[i,None].T
        scores_in_test_one = scores_in_test[0:self.ic.num_boxes_one]
        scores_in_test_two = scores_in_test[(self.ic.num_boxes_one+1):]
        scores_out_test_one = self.ic.scores_out_test[i,None].T
        # Extract the conformity scores for the inliers
        scores_in_cal_one = self.ic.scores_in_calib_one
        scores_in_cal_two = self.ic.scores_in_calib_two
        scores_outin_cal_one = self.ic.scores_outin_calib_one
        # Combine the calibration and test scores
        scores_in_caltest_one = np.concatenate([scores_in_cal_one, scores_in_test_one],1).T
        scores_in_caltest_two = np.concatenate([scores_in_cal_two, scores_in_test_two],1).T
        scores_outin_caltest_one = np.concatenate([scores_outin_cal_one, scores_out_test_one],1).T
        # Pick a new test point (j-th element)
        new_scores_in_test_one = scores_in_caltest_one[j]
        new_scores_in_test_two = scores_in_caltest_two[j]
        new_scores_out_test_one = scores_outin_caltest_one[j]
        # Pick the new calibration scores
        new_scores_in_cal_one = np.concatenate([scores_in_caltest_one[:j],scores_in_caltest_one[(j+1):]],0).T
        new_scores_in_cal_two = np.concatenate([scores_in_caltest_two[:j],scores_in_caltest_two[(j+1):]],0).T
        new_scores_outin_cal_one = np.concatenate([scores_outin_caltest_one[:j],scores_outin_caltest_one[(j+1):]],0).T
        # Make a new copy of the integrative conformal inference method, with new randomly shuffled scores
        new_ic = copy.deepcopy(self.ic)
        new_ic.scores_in_calib_one = new_scores_in_cal_one
        new_ic.scores_in_calib_two = new_scores_in_cal_two
        new_ic.scores_outin_calib_one = new_scores_outin_cal_one
        #new_ic.scores_in_test = np.concatenate([new_scores_in_test_one,new_scores_in_test_two],0)
        #new_ic.scores_out_test = new_scores_out_test_one
        # Compute conformal p-values with the new perturbed scores
        pvals = np.zeros((n_test,))
        for k in range(n_test):
            if k != i:
                pvals[k], _, _ = new_ic._compute_pvalue(new_ic.scores_in_test[k], new_ic.scores_out_test[k])
        reject, _, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
        R = np.sum(reject)
        return R

    def _estimate_R_tilde(self, i, alpha, estimate='median'):
        n_test = self.ic.scores_in_test.shape[0]
        R_tilde_tmp = -np.ones((n_test,))
        for j in range(n_test):
            R_tilde_tmp[j] = self._estimate_R_tilde_single(i, j, alpha)
        if estimate=='median':
            R_tilde = np.median(R_tilde_tmp)
        elif estimate=='min':
            R_tilde = np.min(R_tilde_tmp)
        else:
            print("Error: unknown estimation method {:s}".format(estimate))
        return R_tilde

    def filter_fdr_conditional(self, X_test, alpha):
        n = X_test.shape[0]
        # First, compute the integrative conformal p-values
        pvals = self.ic.compute_pvalues(X_test, return_prepvals=False)
        # Extract the conformity scores for the test points
        scores_in_test = self.ic.scores_in_test
        scores_out_test = self.ic.scores_in_test
        # Estimate number of rejections R-tilde for each test point
        R_tilde = -np.ones((n,))
        for i in tqdm(range(n)):
            R_tilde[i] = self._estimate_R_tilde(i, alpha)
        # Define the preliminary rejection set R+
        R_plus = np.where(pvals <= alpha*R_tilde/n)[0]
        if len(R_plus)==0:
            return [], False
        # Check whether we need to prune
        if np.sum([len(R_plus) < R_tilde[i] for i in R_plus]) == 0:
            print("Pruning is not needed.")
            return R_plus, False
        else:
            print("Pruning is needed.")
            epsilon = np.random.uniform(size=(n,))
            pvals_fake = epsilon[R_plus] * R_tilde[R_plus] / len(R_plus)
            rejected_fake, _, _, _ = multipletests(pvals_fake, alpha=alpha, method='fdr_bh')
            R = np.sum(rejected_fake)
            R_list = np.array([i for i in R_plus if epsilon[i] <= R / R_tilde[i]])
            return R_list, True

        return None

    def filter_fdr_bh(self, X_test, alpha):
        n = X_test.shape[0]
        pvals = self.ic.compute_pvalues(X_test, return_prepvals=False)
        rejected, _, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
        return np.where(rejected)[0]
        
