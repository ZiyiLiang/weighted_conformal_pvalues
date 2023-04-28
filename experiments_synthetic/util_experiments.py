import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import pdb

def filter_BH(pvals, alpha, Y):
    is_nonnull = (Y==1)
    reject, pvals_adj, _, _ = multipletests(pvals, alpha, method="fdr_bh")
    rejections = np.sum(reject)
    if rejections>0:
        fdp = 1-np.mean(is_nonnull[np.where(reject)[0]])
        power = np.sum(is_nonnull[np.where(reject)[0]]) / np.sum(is_nonnull)
    else:
        fdp = 0
        power = 0
    return rejections, fdp, power

def filter_StoreyBH(pvals, alpha, Y, lamb=0.5):
    n = len(pvals)
    R = np.sum(pvals<=lamb)
    pi = (1+n-R) / (n*(1.0 - lamb))
    pvals[pvals>lamb] = 1
    return filter_BH(pvals, alpha/pi, Y)

def filter_fixed(pvals, alpha, Y):
    is_nonnull = (Y==1)
    reject = (pvals<=alpha)
    rejections = np.sum(reject)
    if rejections>0:
        if np.sum(Y==0)>0:
            fpr = np.mean(reject[np.where(Y==0)[0]])
        else:
            fpr = 0
        if np.sum(Y==1)>0:
            tpr = np.mean(reject[np.where(Y==1)[0]])
        else:
            tpr = 0
    else:
        fpr = 0
        tpr = 0
    return rejections, fpr, tpr

def eval_pvalues(pvals, Y, alpha_list):
    # Evaluate with BH and Storey-BH
    fdp_list = -np.ones((len(alpha_list),1))
    power_list = -np.ones((len(alpha_list),1))
    rejections_list = -np.ones((len(alpha_list),1))
    fdp_storey_list = -np.ones((len(alpha_list),1))
    power_storey_list = -np.ones((len(alpha_list),1))
    rejections_storey_list = -np.ones((len(alpha_list),1))
    for alpha_idx in range(len(alpha_list)):
        alpha = alpha_list[alpha_idx]
        rejections_list[alpha_idx], fdp_list[alpha_idx], power_list[alpha_idx] = filter_BH(pvals, alpha, Y)
        rejections_storey_list[alpha_idx], fdp_storey_list[alpha_idx], power_storey_list[alpha_idx] = filter_StoreyBH(pvals, alpha, Y)
    results_tmp = pd.DataFrame({})
    results_tmp["Alpha"] = alpha_list
    results_tmp["BH-Rejections"] = rejections_list
    results_tmp["BH-FDP"] = fdp_list
    results_tmp["BH-Power"] = power_list
    results_tmp["Storey-BH-Rejections"] = rejections_storey_list
    results_tmp["Storey-BH-FDP"] = fdp_storey_list
    results_tmp["Storey-BH-Power"] = power_storey_list
    # Evaluate with fixed threshold
    fpr_list = -np.ones((len(alpha_list),1))
    tpr_list = -np.ones((len(alpha_list),1))
    rejections_list = -np.ones((len(alpha_list),1))
    for alpha_idx in range(len(alpha_list)):
        alpha = alpha_list[alpha_idx]
        rejections_list[alpha_idx], fpr_list[alpha_idx], tpr_list[alpha_idx] = filter_fixed(pvals, alpha, Y)
    results_tmp["Fixed-Rejections"] = rejections_list
    results_tmp["Fixed-FPR"] = fpr_list
    results_tmp["Fixed-TPR"] = tpr_list
    return results_tmp



def sample_beta_mixture(n, u1_ref, nu, gamma):
    mix_id = np.random.binomial(1, gamma, size=(n,))
    idx_ref = np.where(mix_id==1)[0]
    idx_beta = np.where(mix_id==0)[0]
    out = -np.ones((n,))
    if len(idx_ref)>0:
        out[idx_ref] = np.random.choice(u1_ref, len(idx_ref), replace=True)
    if len(idx_beta)>0:
        out[idx_beta] = np.random.beta(nu, 1, size=(len(idx_beta),))
    return out

def estimate_beta_mixture(u1_test, u1_ref):
    # Pre-compute sufficient stats
    u1_test_m1 = np.mean(u1_test)
    u1_test_m2 = np.mean(u1_test**2)
    u1_ref_m1 = np.mean(u1_ref)
    u1_ref_m2 = np.mean(u1_ref**2)

    def eval_gamma(nu):
        gamma = (u1_test_m2 - nu/(nu+2.0)) / (u1_ref_m2 - nu/(nu+2.0))
        return np.clip(gamma, 0, 1)

    def eval_nu(gamma):
        nu = (u1_test_m1 - gamma * u1_ref_m1)
        nu /= ( 1 - gamma - (u1_test_m1 - gamma * u1_ref_m1))
        return np.maximum(nu, 0.0001)

    def update_nu(nu):
        nu2 = (nu+1)*(nu+2) * (u1_test_m1 - nu/(nu+1)) * (u1_ref_m2 - nu/(nu+2))
        nu2 -= (nu+1)*(nu+2) * u1_ref_m1 * u1_test_m2
        nu2 += nu*(nu+1) * u1_ref_m1 + nu*(nu+2) * u1_test_m2
        nu = np.sqrt(np.maximum(nu2,0.0001))
        return nu


    # Initialize
    gamma = 0.5
    nu = eval_nu(gamma)
    print("It {:3d}: nu = {:.3f}, gamma = {:.3f}".format(0, nu, gamma))
    # Iterate until convergence
    nu_old = nu
    gamma_old = gamma
    converged = False
    it = 0
    while (not converged) and (it < 100):
        it = it + 1
        nu = update_nu(nu)
        gamma = eval_gamma(nu)
        print("It {:3d}: nu = {:.3f}, gamma = {:.3f}".format(it, nu, gamma))
        delta_nu = np.abs(nu - nu_old) / (nu_old+1e-6)
        delta_gamma = np.abs(gamma - gamma_old) / (gamma_old+1e-6)
        nu_old = nu
        gamma_old = gamma
        if np.maximum(delta_nu, delta_gamma) < 1e-6:
            converged = True

    # Check whether method converged safely
    if gamma==1:
        gamma = np.nan
        nu = np.nan

    return nu, gamma
