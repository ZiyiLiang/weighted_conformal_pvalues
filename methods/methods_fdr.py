import numpy as np
import copy
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

class IntegrativeConformalFDR:
    '''FDR control with conditional calibration'''
    def __init__(self, ic):
        '''
        Parameter:
        ----------
        ic:     class object
                the IntegrativeConformal class
        '''
        self.ic = ic

    def _estimate_R_tilde_single(self, i, j, alpha):
        n_test = self.ic.scores_in_test.shape[0]
        n_cal = self.ic.scores_in_calib_one.shape[1]
        assert( (j>=-1) and (j<=n_cal))
        # Extract the conformity scores for the i-th test point
        scores_in_test = self.ic.scores_in_test[i,None].T
        scores_in_test_one = scores_in_test[0:self.ic.num_boxes_one]
        scores_in_test_two = scores_in_test[self.ic.num_boxes_one:]
        scores_out_test_one = (self.ic.scores_out_test[i,None].T)[0:self.ic.num_boxes_one]
        # Extract the conformity scores for the inliers
        scores_in_cal_one = self.ic.scores_in_calib_one
        scores_in_cal_two = self.ic.scores_in_calib_two
        scores_outin_cal_one = self.ic.scores_outin_calib_one
        # Combine the calibration and test scores
        scores_in_caltest_one = np.concatenate([scores_in_cal_one, scores_in_test_one],1).T
        scores_in_caltest_two = np.concatenate([scores_in_cal_two, scores_in_test_two],1).T
        scores_outin_caltest_one = np.concatenate([scores_outin_cal_one, scores_out_test_one],1).T
        # Pick the new calibration scores, select all if j=-1
        if j == -1:
            new_scores_in_cal_one = scores_in_caltest_one.T
            new_scores_in_cal_two = scores_in_caltest_two.T
            new_scores_outin_cal_one = scores_outin_caltest_one.T
        # Else leave the jth score out
        else:
            new_scores_in_cal_one = np.concatenate([scores_in_caltest_one[:j],scores_in_caltest_one[(j+1):]],0).T
            new_scores_in_cal_two = np.concatenate([scores_in_caltest_two[:j],scores_in_caltest_two[(j+1):]],0).T
            new_scores_outin_cal_one = np.concatenate([scores_outin_caltest_one[:j],scores_outin_caltest_one[(j+1):]],0).T

        # Make a new copy of the integrative conformal inference method, with new randomly shuffled scores
        new_ic = copy.deepcopy(self.ic)
        new_ic.scores_in_calib_one = new_scores_in_cal_one
        new_ic.scores_in_calib_two = new_scores_in_cal_two
        new_ic.scores_outin_calib_one = new_scores_outin_cal_one
        # Compute conformal p-values with the new perturbed scores, the ith p-value is set to 0
        pvals = np.zeros((n_test,))
        for k in range(n_test):
            if k != i:
                pvals[k], _, _ = new_ic._compute_pvalue(new_ic.scores_in_test[k], new_ic.scores_out_test[k])
        reject, _, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
        R = np.sum(reject)
        return R

    def _estimate_R_tilde(self, i, alpha, J_max=None, loo='median'):
        if loo=='none':
            R_tilde = self._estimate_R_tilde_single(i, -1, alpha)
        else:
            n_cal = self.ic.scores_in_calib_one.shape[1]
            n_test = self.ic.scores_in_test.shape[0]
            # Set maximum number of leave-one-out points
            if J_max is None:
                J_max = n_test
            else:
                J_max = np.minimum(J_max,n_cal+1)
            R_tilde_tmp = -np.ones((J_max,))
            j_seq = np.random.choice(n_cal+1, size=J_max, replace=False)
            # Compute the conformal p-values with leave-one-out calibration set
            for j in range(J_max):
                R_tilde_tmp[j] = self._estimate_R_tilde_single(i, j_seq[j], alpha)
            if loo=='median':
                R_tilde = np.median(R_tilde_tmp)
            elif loo=='min':
                R_tilde = np.min(R_tilde_tmp)
            else:
                print("Error: unknown estimation method {:s}".format(loo))
        return R_tilde

    def filter_fdr_conditional(self, X_test, alpha, J_max=None, loo='median', verbose=True):
        '''Compute the rejection list by the conditional calibration method
        
        Parameters:
        -----------
        X_test:     array_like
                    test points
        alpha:      float
                    user specified FDR level
        J_max:      int
                    maximum number of leave-one-out points, if not specified, it will be set to the number of test points
        loo:        str
                    needs to be either 'median' or 'min', 'median' will use the median number of leave-one-out R-tildes
                    whereas 'min' uses the minimum.
        
        Returns:
        --------
        2 returns, first one is a ndarray of booleans, true if the hypothesis is rejected;
                   second one is a boolean indicating if the rejection is pruned or not.
        '''
        n = X_test.shape[0]
        # First, compute the integrative conformal p-values
        pvals = self.ic.compute_pvalues(X_test, return_prepvals=False)
        # Estimate number of rejections R-tilde for each test point
        R_tilde = -np.ones((n,))
        for i in tqdm(range(n)):
            R_tilde[i] = self._estimate_R_tilde(i, alpha, J_max=J_max, loo=loo)
        # Define the preliminary rejection set R+
        R_plus_list = (pvals <= alpha*R_tilde/n)
        R_plus = np.where(R_plus_list)[0]
        if len(R_plus)==0:
            return np.array([False]*n), False
        # Check whether we need to prune
        if np.sum([len(R_plus) < R_tilde[i] for i in R_plus]) == 0:
            if verbose:
                print("Pruning is not needed.")
            return R_plus_list, False
        else:
            if verbose:
                print("Pruning is needed.")
            epsilon = np.random.uniform(size=(n,))
            pvals_fake = epsilon[R_plus] * R_tilde[R_plus] / len(R_plus)
            rejected_fake, _, _, _ = multipletests(pvals_fake, alpha=alpha, method='fdr_bh')
            R = np.sum(rejected_fake)
            R_list_support = np.array([i for i in R_plus if epsilon[i] <= R / R_tilde[i]])
            R_list = np.array([False]*n)
            if len(R_list_support)>0:
                R_list[R_list_support] = True
            return R_list, True

        return None

    def filter_fdr_bh(self, X_test, alpha):
        '''Compute the rejection list by the BH method
        
        Parameters:
        -----------
        X_test:     array_like
                    test points
        alpha:      float
                    user specified FDR level

        Returns:
        --------
        rejected:   ndarray, boolean
                    true for hypothesis that can be rejected for given alpha
        '''
        n = X_test.shape[0]
        pvals = self.ic.compute_pvalues(X_test, return_prepvals=False)
        rejected, _, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
        return rejected

    def filter_fdr_by(self, X_test, alpha):
        '''Compute the rejection list by the BY method
        
        Parameters:
        -----------
        X_test:     array_like
                    test points
        alpha:      float
                    user specified FDR level

        Returns:
        --------
        rejected:   ndarray, boolean
                    true for hypothesis that can be rejected for given alpha
        '''
        n = X_test.shape[0]
        pvals = self.ic.compute_pvalues(X_test, return_prepvals=False)
        rejected, _, _, _ = multipletests(pvals, alpha=alpha, method='fdr_by')
        return rejected
        
