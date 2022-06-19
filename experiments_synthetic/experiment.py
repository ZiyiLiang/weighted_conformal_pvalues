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
from methods_split import BinaryConformal, OneClassConformal

#########################
# Experiment parameters #
#########################

if False: # Input parameters
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
    a = 0.9
    purity = 0.8
    random_state = 1


# Fixed experiment parameters
n_test = 1000
purity_test = 0.5
calib_size = 0.5
alpha_list = [0.05, 0.1]

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
    'IF':IsolationForest(contamination=0.1,random_state=random_state),
    'SVM': OneClassSVM(nu=0.1, kernel="rbf"),
    'LOF': LocalOutlierFactor(contamination=0.1, novelty=True)
}



###############
# Output file #
###############
outfile_prefix = "results/setup" + str(setup) + "/" +str(data_name) + "_n"+str(n) + "_p" + str(p) + "_a" + str(a) + "_purity" + str(purity) + "_seed" + str(random_state)
print("Output file: {:s}.".format(outfile_prefix), end="\n")

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
    def __init__(self, data_name):
        if data_name=="circles":
            self.model = ConcentricCircles(p, a)
        elif data_name=="binomial":
            self.model = BinomialModel(p, a)
        else:
            print("Error: unknown model name!")
            exit(0)

    def sample(self, n, purity, random_state=2022):
        return self.model.sample(n, purity, random_state=random_state)

dataset = DataSet(data_name)

###################
# Run experiments #
###################

def apply_BH(pvals, alpha, Y):
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

def eval_pvalues(pvals, Y):
    fdp_list = -np.ones((len(alpha_list),1))
    power_list = -np.ones((len(alpha_list),1))
    for alpha_idx in range(len(alpha_list)):
        alpha = alpha_list[alpha_idx]
        fdp_list[alpha_idx], power_list[alpha_idx] = apply_BH(pvals, alpha, Y)
    new_results = pd.DataFrame({})
    new_results["Alpha"] = alpha_list
    new_results["FDP"] = fdp_list
    new_results["Power"] = power_list
    return new_results

    
def run_experiment(random_state):
    # Sample the training/calibration data
    X, Y = dataset.sample(n, purity)
    X_in = X[Y==0]
    X_out = X[Y==1]
    # Sample the test data
    X_test, Y_test = dataset.sample(n_test, purity_test)

    # Initialize result data frame
    results = pd.DataFrame({})

    # Compute conformal p-values via binary classification
    print("Running {:d} binary classifiers...".format(len(binary_classifiers)))
    sys.stdout.flush()
    for bc_name in tqdm(binary_classifiers.keys()):
        bc = binary_classifiers[bc_name]
        method = BinaryConformal(X_in, X_out, bc, calib_size=calib_size, verbose=False)
        pvals_test = method.compute_pvalues(X_test)
        new_results = eval_pvalues(pvals_test, Y_test)
        new_results["Method"] = "Binary"
        new_results["Model"] = bc_name
        results = pd.concat([results, new_results])

    # Compute conformal p-values via one-class classification
    print("Running {:d} one-class classifiers...".format(len(binary_classifiers)))
    sys.stdout.flush()
    for occ_name in tqdm(oneclass_classifiers.keys()):
        occ = oneclass_classifiers[occ_name]
        method = OneClassConformal(X_in, occ, calib_size=calib_size)
        pvals_test = method.compute_pvalues(X_test)
        new_results = eval_pvalues(pvals_test, Y_test)
        new_results["Method"] = "One-class"
        new_results["Model"] = occ_name
        results = pd.concat([results, new_results])   

    # Continue from here

    return results

results = run_experiment(random_state)
results = add_header(results)

pdb.set_trace()

################
# Save results #
################
if False:
    outfile = outfile_prefix + ".txt"
    res.to_csv(outfile, index=False)
    print("\nResults written to {:s}\n".format(outfile))
    sys.stdout.flush()
