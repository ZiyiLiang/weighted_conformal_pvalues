import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import pdb

# Binary classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import os, sys
sys.path.append("../methods")

from models import GaussianMixture, ConcentricCircles, BinomialModel
from methods_split import BinaryConformal, OneClassConformal, WeightedOneClassConformal

#########################
# Experiment parameters #
#########################

if True: # Input parameters
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    model_num = 1
    if len(sys.argv) != 8:
        print("Error: incorrect number of parameters.")
        quit()

    setup = int(sys.argv[1])
    data_name = sys.argv[2]
    n = int(sys.argv[3])
    p = int(sys.argv[4])
    a = float(sys.argv[5])
    purity = float(sys.argv[6])
    random_state = int(sys.argv[7])

else: # Default parameters
    setup = 1
    data_name = "circles"
    n = 1000
    p = 100
    a = 0.7
    purity = 0.7
    random_state = 2022


# Fixed experiment parameters
n_test = 1000
purity_test = 0.5
calib_size = 0.5
alpha_list = [0.01, 0.02, 0.05, 0.1, 0.2]
num_repetitions = 2

# Define list of possible two-class classifiers with desired hyper-parameters
binary_classifiers = {
    'RF': RandomForestClassifier(max_depth=2, random_state=random_state),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVC': SVC(kernel="linear", C=0.025, probability=True),
    'NB' : GaussianNB(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'MLP': MLPClassifier(max_iter=500, random_state=random_state)
}

# List of possible one-class classifiers with desired hyper-parameters
oneclass_classifiers = {
    'IF': IsolationForest(contamination=0.1,random_state=random_state),
    'SVM': OneClassSVM(nu=0.1, kernel="rbf"),
    'LOF': LocalOutlierFactor(contamination=0.1, novelty=True)
}


###############
# Output file #
###############
outfile_prefix = "results/setup" + str(setup) + "/" +str(data_name) + "_n"+str(n) + "_p" + str(p) + "_a" + str(a) + "_purity" + str(purity) + "_seed" + str(random_state)
outfile = outfile_prefix + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

# Header for results file
def add_header(df):
    df["Setup"] = setup
    df["Data"] = data_name
    df["n"] = n
    df["p"] = p
    df["Signal"] = a
    df["Purity"] = purity
    df["Seed"] = random_state
    return df

#########################
# Data-generating model #
#########################

class DataSet:
    def __init__(self, data_name, random_state=None):
        if data_name=="circles":
            self.model = ConcentricCircles(p, a, random_state=random_state)
        elif data_name=="binomial":
            self.model = BinomialModel(p, a, random_state=random_state)
        else:
            print("Error: unknown model name!")
            exit(0)

    def sample(self, n, purity):
        return self.model.sample(n, purity)

###################
# Run experiments #
###################

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
    return fdp, power

def filter_fixed(pvals, alpha, Y):
    is_nonnull = (Y==1)
    reject = np.where(pvals<=alpha)[0]
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
    return fpr, tpr

def eval_pvalues(pvals, Y):
    # Evaluate with BH
    fdp_list = -np.ones((len(alpha_list),1))
    power_list = -np.ones((len(alpha_list),1))
    for alpha_idx in range(len(alpha_list)):
        alpha = alpha_list[alpha_idx]
        fdp_list[alpha_idx], power_list[alpha_idx] = filter_BH(pvals, alpha, Y)
    results_tmp = pd.DataFrame({})
    results_tmp["Alpha"] = alpha_list
    results_tmp["BH-FDP"] = fdp_list
    results_tmp["BH-Power"] = power_list
    # Evaluate with fixed threshold
    fpr_list = -np.ones((len(alpha_list),1))
    tpr_list = -np.ones((len(alpha_list),1))
    for alpha_idx in range(len(alpha_list)):
        alpha = alpha_list[alpha_idx]        
        fpr_list[alpha_idx], tpr_list[alpha_idx] = filter_BH(pvals, alpha, Y)
    results_tmp["Fixed-FPR"] = fpr_list
    results_tmp["Fixed-TPR"] = tpr_list
    return results_tmp

    
def run_experiment(dataset, random_state):
    # Sample the training/calibration data
    X, Y = dataset.sample(n, purity)
    X_in = X[Y==0]
    X_out = X[Y==1]
    # Sample the test data
    X_test, Y_test = dataset.sample(n_test, purity_test)

    # Initialize result data frame
    results = pd.DataFrame({})

    # Conformal p-values via binary classification
    print("Running {:d} binary classifiers...".format(len(binary_classifiers)))
    sys.stdout.flush()
    for bc_name in tqdm(binary_classifiers.keys()):
        bc = binary_classifiers[bc_name]
        method = BinaryConformal(X_in, X_out, bc, calib_size=calib_size, verbose=False)
        pvals_test = method.compute_pvalues(X_test)
        results_tmp = eval_pvalues(pvals_test, Y_test)
        results_tmp["Method"] = "Binary"
        results_tmp["Model"] = bc_name
        results = pd.concat([results, results_tmp])

    # Conformal p-values via one-class classification
    print("Running {:d} one-class classifiers...".format(len(oneclass_classifiers)))
    sys.stdout.flush()
    for occ_name in tqdm(oneclass_classifiers.keys()):
        occ = oneclass_classifiers[occ_name]
        method = OneClassConformal(X_in, occ, calib_size=calib_size, verbose=False)
        pvals_test = method.compute_pvalues(X_test)
        results_tmp = eval_pvalues(pvals_test, Y_test)
        results_tmp["Method"] = "One-Class"
        results_tmp["Model"] = occ_name
        results = pd.concat([results, results_tmp])   

    ## Conformal p-values via weighted one-class classification
    print("Running {:d} weighted one-class classifiers...".format(len(oneclass_classifiers)))
    sys.stdout.flush()
    for occ_name in tqdm(oneclass_classifiers.keys()):
        occ = oneclass_classifiers[occ_name]
        method = WeightedOneClassConformal(X_in, X_out, bboxes_one=[occ], calib_size=calib_size, tuning=True, progress=False, verbose=False)
        pvals_test = method.compute_pvalues(X_test)
        results_tmp = eval_pvalues(pvals_test, Y_test)
        results_tmp["Method"] = "Weighted One-Class"
        results_tmp["Model"] = occ_name
        results = pd.concat([results, results_tmp])   

    ## Conformal p-values via weighted one-class classification and learning ensemble
    print("Running weighted one-class classifier with learning ensemble...")
    sys.stdout.flush()
    bboxes_one = list(oneclass_classifiers.values())
    bboxes_two = list(binary_classifiers.values())
    method = WeightedOneClassConformal(X_in, X_out, 
                                       bboxes_one=bboxes_one, bboxes_two=bboxes_two,
                                       calib_size=calib_size, tuning=True, progress=True, verbose=False)
    pvals_test = method.compute_pvalues(X_test)
    results_tmp = eval_pvalues(pvals_test, Y_test)
    results_tmp["Method"] = "Ensemble (weighted)"
    results_tmp["Model"] = "Ensemble"
    results = pd.concat([results, results_tmp])   
        
    ## Conformal p-values via learning ensemble (no weighting)
    print("Running weighted one-class classifier with learning ensemble (without weighting)...")
    sys.stdout.flush()
    bboxes_one = list(oneclass_classifiers.values())
    bboxes_two = list(binary_classifiers.values())
    method = WeightedOneClassConformal(X_in, X_out, 
                                       bboxes_one=bboxes_one, bboxes_two=bboxes_two,
                                       calib_size=calib_size, ratio=False, tuning=True, progress=True, verbose=False)
    pvals_test = method.compute_pvalues(X_test)
    results_tmp = eval_pvalues(pvals_test, Y_test)
    results_tmp["Method"] = "Ensemble"
    results_tmp["Model"] = "Ensemble"
    results = pd.concat([results, results_tmp])   

    # Continue from here

    return results

# Initialize result data frame
results = pd.DataFrame({})

for r in range(num_repetitions):
    print("\nStarting repetition {:d} of {:d}:\n".format(r+1, num_repetitions))
    sys.stdout.flush()
    # Change random seed for this repetition
    random_state_new = 10*num_repetitions*random_state + r
    dataset = DataSet(data_name, random_state=random_state_new)
    # Run experiment and collect results
    results_new = run_experiment(dataset, random_state_new)
    results_new = add_header(results_new)
    results_new["Repetition"] = r
    results = pd.concat([results, results_new])   
    # Save results
    results.to_csv(outfile, index=False)
    print("\nResults written to {:s}\n".format(outfile))
    sys.stdout.flush()
    
print("\nAll experiments completed.\n")
sys.stdout.flush()
