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
