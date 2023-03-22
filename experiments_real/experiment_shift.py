import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from scipy.io import arff, loadmat
from sklearn.model_selection import train_test_split
import pdb

from sklearn.datasets import load_digits, fetch_covtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import re
import pickle

# Binary classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM

import os, sys
sys.path.append("../methods")

from models import GaussianMixture, ConcentricCircles, ConcentricCirclesMixture, BinomialModel
from methods_split import BinaryConformal, OneClassConformal, IntegrativeConformal

from util_experiments import eval_pvalues
from utils_data_shift import DataSet

#########################
# Experiment parameters #
#########################

if True: # Input parameters
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    model_num = 1
    if len(sys.argv) != 5:
        print("Error: incorrect number of parameters.")
        quit()
    data_name = sys.argv[1]
    n_in = int(sys.argv[2])
    n_out = int(sys.argv[3])
    random_state = int(sys.argv[4])

else: # Default parameters
    data_name = "cifar-10"
    n_in = 1000
    n_out = 10
    random_state = 2022


# Fixed experiment parameters
outlier_shift = 0.5
calib_size = 0.5
n_test = 1000
alpha_list = [0.01, 0.02, 0.05, 0.1, 0.2]
num_repetitions = 1

if data_name=="images_flowers":
    prop_mix = 0.1
elif data_name=="images_animals":
    prop_mix = 0.1
elif data_name=="images_cars":
    prop_mix = 0.1
elif data_name=="covtype":
    prop_mix = 0.1
elif data_name=="mammography":
    prop_mix = 0.1
else:
    prop_mix = 0.1

# List of possible one-class classifiers with desired hyper-parameters
oneclass_classifiers = {
    'SVM-rbf': OneClassSVM(kernel='rbf'),
    'SVM-sig': OneClassSVM(kernel='sigmoid'),
    'SVM-pol': OneClassSVM(kernel='poly', degree=3),
    'IF': IsolationForest(random_state=random_state),
    'LOF': LocalOutlierFactor(novelty=True),
    'SVM-sgd': SGDOneClassSVM(random_state=random_state)
}

# Define list of possible two-class classifiers with desired hyper-parameters
binary_classifiers = {
    'RF': RandomForestClassifier(random_state=random_state),
    'KNN': KNeighborsClassifier(),
    'SVC': SVC(probability=True),
    'NB' : GaussianNB(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,100,), random_state=random_state)
}

#########################
# Data-generating model #
#########################

base_path = "../experiments_real/data/"
dataset = DataSet(base_path, data_name, random_state=0, prop_mix=prop_mix, outlier_shift=outlier_shift)

n_in = np.minimum(n_in, dataset.n_in)
n_out = np.minimum(n_out, dataset.n_out)

###############
# Output file #
###############
outfile_prefix = "results_shift/" + str(data_name) + "_nin"+str(n_in) + "_nout"+str(n_out) + "_seed" + str(random_state)
outfile = outfile_prefix + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

if os.path.exists(outfile):
    print("Output file found. Quitting!")
#    exit(0)

# Header for results file
def add_header(df):
    df["Data"] = data_name
    df["n_in"] = n_in
    df["n_out"] = n_out
    df["Seed"] = random_state
    return df

###################
# Run experiments #
###################

def run_experiment(dataset, random_state):
    # Sample the test data
    X_test, Y_test = dataset.sample_test(n=n_test)

    # Sample the training/calibration data
    X, Y = dataset.sample(n_in=n_in, n_out=n_out)
    X_in = X[Y==0]
    X_out = X[Y==1]

    print("--------------------")
    print("Number of features: {}.".format(X.shape[1]))
    print("Number of inliers in training/calibration data: {}.".format(np.sum(Y==0)))
    print("Number of outliers in training/calibration data: {}.".format(np.sum(Y==1)))

    print("Number of inliers in test data: {}.".format(np.sum(Y_test==0)))
    print("Number of outliers in test data: {}.".format(np.sum(Y_test==1)))
    print("--------------------")

    # Initialize result data frame
    results = pd.DataFrame({})

    # Conformal p-values via binary classification
    print("Running {:d} binary classifiers...".format(len(binary_classifiers)))
    sys.stdout.flush()
    for bc_name in tqdm(binary_classifiers.keys()):
        bc = binary_classifiers[bc_name]
        method = BinaryConformal(X_in, X_out, bc, calib_size=calib_size, verbose=False)
        pvals_test = method.compute_pvalues(X_test)
        results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
        results_tmp["Method"] = "Binary"
        results_tmp["Model"] = bc_name
        results_tmp["E_U1_Y0"] = np.nan
        results_tmp["1/log(n1+1)"] = np.nan
        results = pd.concat([results, results_tmp])

    # Conformal p-values via one-class classification
    print("Running {:d} one-class classifiers...".format(len(oneclass_classifiers)))
    sys.stdout.flush()
    for occ_name in tqdm(oneclass_classifiers.keys()):
        occ = oneclass_classifiers[occ_name]
        method = OneClassConformal(X_in, occ, calib_size=calib_size, verbose=False)
        pvals_test = method.compute_pvalues(X_test)
        results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
        results_tmp["Method"] = "One-Class"
        results_tmp["Model"] = occ_name
        results_tmp["E_U1_Y0"] = np.nan
        results_tmp["1/log(n1+1)"] = np.nan
        results = pd.concat([results, results_tmp])

    ## Conformal p-values via weighted one-class classification and learning ensemble
    print("Running weighted classifiers with learning ensemble...")
    sys.stdout.flush()
    bboxes_one = list(oneclass_classifiers.values())
    bboxes_two = list(binary_classifiers.values())
    method = IntegrativeConformal(X_in, X_out,
                                       bboxes_one=bboxes_one, bboxes_two=bboxes_two,
                                       calib_size=calib_size, tuning=True, progress=True, verbose=False)
    pvals_test, pvals_test_0, pvals_test_1 = method.compute_pvalues(X_test, return_prepvals=True)
    results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    results_tmp["Method"] = "Ensemble"
    results_tmp["Model"] = "Ensemble"
    results_tmp["E_U1_Y0"] = np.mean(pvals_test_1)
    results_tmp["1/log(n1+1)"] = 1/np.log(int(X_out.shape[0]*calib_size)+1.0)
    results = pd.concat([results, results_tmp])

    ## Conformal p-values via weighted one-class learning ensemble
    print("Running weighted classifiers with learning ensemble...")
    sys.stdout.flush()
    bboxes_one = list(oneclass_classifiers.values())
    bboxes_two = list(binary_classifiers.values())
    method = IntegrativeConformal(X_in, X_out,
                                       bboxes_one=bboxes_one,
                                       calib_size=calib_size, tuning=True, progress=True, verbose=False)
    pvals_test, pvals_test_0, pvals_test_1 = method.compute_pvalues(X_test, return_prepvals=True)
    results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    results_tmp["Method"] = "Ensemble (one-class)"
    results_tmp["Model"] = "Ensemble"
    results_tmp["E_U1_Y0"] = np.mean(pvals_test_1)
    results_tmp["1/log(n1+1)"] = 1/np.log(int(X_out.shape[0]*calib_size)+1.0)
    results = pd.concat([results, results_tmp])

    ## Conformal p-values via learning ensemble (no weighting)
    print("Running weighted classifiers with learning ensemble (without weighting)...")
    sys.stdout.flush()
    bboxes_one = list(oneclass_classifiers.values())
    bboxes_two = list(binary_classifiers.values())
    method = IntegrativeConformal(X_in, X_out,
                                  bboxes_one=bboxes_one, bboxes_two=bboxes_two,
                                  calib_size=calib_size, ratio=False, tuning=True, progress=True, verbose=False)
    pvals_test = method.compute_pvalues(X_test)
    results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    results_tmp["Method"] = "Ensemble (mixed, unweighted)"
    results_tmp["Model"] = "Ensemble"
    results_tmp["E_U1_Y0"] = np.nan
    results_tmp["1/log(n1+1)"] = np.nan
    results = pd.concat([results, results_tmp])

    ## Conformal p-values via learning ensemble (one-class, no weighting)
    print("Running weighted classifiers with learning ensemble (one-class, without weighting)...")
    sys.stdout.flush()
    bboxes_one = list(oneclass_classifiers.values())
    bboxes_two = list(binary_classifiers.values())
    method = IntegrativeConformal(X_in, X_out,
                                  bboxes_one=bboxes_one,
                                  calib_size=calib_size, ratio=False, tuning=True, progress=True, verbose=False)
    pvals_test = method.compute_pvalues(X_test)
    results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    results_tmp["Method"] = "Ensemble (one-class, unweighted)"
    results_tmp["Model"] = "Ensemble"
    results_tmp["E_U1_Y0"] = np.nan
    results_tmp["1/log(n1+1)"] = np.nan
    results = pd.concat([results, results_tmp])

    # ## Conformal p-values via binary ensemble (no weighting)
    # print("Running binary classifiers with learning ensemble (without weighting)...")
    # sys.stdout.flush()
    # bboxes_one = list(oneclass_classifiers.values())
    # bboxes_two = list(binary_classifiers.values())
    # method = IntegrativeConformal(X_in, X_out,
    #                                    bboxes_two=bboxes_two,
    #                                    calib_size=calib_size, ratio=False, tuning=True, progress=True, verbose=False)
    # pvals_test = method.compute_pvalues(X_test)
    # results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    # results_tmp["Method"] = "Ensemble (binary, unweighted)"
    # results_tmp["Model"] = "Ensemble"
    # results_tmp["E_U1_Y0"] = np.nan
    # results_tmp["1/log(n1+1)"] = np.nan
    # results = pd.concat([results, results_tmp])

    return results

# Initialize result data frame
results = pd.DataFrame({})

for r in range(num_repetitions):
    print("\nStarting repetition {:d} of {:d}:\n".format(r+1, num_repetitions))
    sys.stdout.flush()
    # Change random seed for this repetition
    random_state_new = 10*num_repetitions*random_state + r
    dataset = DataSet(base_path, data_name, random_state=random_state_new, prop_mix=prop_mix, outlier_shift=outlier_shift)
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
