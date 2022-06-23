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

from models import GaussianMixture, ConcentricCircles, ConcentricCirclesMixture, BinomialModel
from methods_split import BinaryConformal, OneClassConformal, IntegrativeConformal

from util_experiments import eval_pvalues

#########################
# Data-generating model #
#########################

class DataSet:
    def __init__(self, data_name, random_state=None):
        if data_name=="circles":
            self.model = ConcentricCircles(p, a, random_state=random_state)
        elif data_name=="circles-mixed":
            self.model = ConcentricCirclesMixture(p, a, random_state=random_state)
        elif data_name=="binomial":
            self.model = BinomialModel(p, a, random_state=random_state)
        else:
            print("Error: unknown model name!")
            exit(0)

    def sample(self, n, purity):
        return self.model.sample(n, purity)

#########################
# Experiment parameters #
#########################

if True: # Input parameters
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    model_num = 1
    if len(sys.argv) != 9:
        print("Error: incorrect number of parameters.")
        quit()

    setup = int(sys.argv[1])
    data_name = sys.argv[2]
    n = int(sys.argv[3])
    p = int(sys.argv[4])
    a = float(sys.argv[5])
    purity = float(sys.argv[6])
    num_models = int(sys.argv[7])
    random_state = int(sys.argv[8])

else: # Default parameters
    setup = 1
    data_name = "circles-mixed"
    n = 500
    p = 1000
    a = 0.7
    purity = 0.8
    num_models = 2
    random_state = 2022

# Fixed experiment parameters
n_test = 1000
purity_test = 0.5
calib_size = 0.5
alpha_list = [0.01, 0.02, 0.05, 0.1, 0.2]
num_repetitions = 1

# List of possible one-class classifiers with desired hyper-parameters
dict_oneclass_classifiers = {
    'LOF': LocalOutlierFactor(novelty=True),
    'SVM-pol': OneClassSVM(kernel='poly', degree=3),
    'SVM-sig': OneClassSVM(kernel='sigmoid', degree=3),
    'SVM-rbf': OneClassSVM(kernel='rbf', degree=3)
}

# Define list of possible two-class classifiers with desired hyper-parameters
dict_binary_classifiers = {
    'KNN': KNeighborsClassifier(),
    'SVC': SVC(probability=True),
    'NB' : GaussianNB(),
    'MLP': MLPClassifier(max_iter=500, random_state=random_state)
}

# List of one-class classifiers for this experiment
oneclass_classifiers = dict()
num_models_standard = np.minimum(4, num_models)
for s in range(num_models_standard):
    box_name = list(dict_oneclass_classifiers.keys())[s]
    oneclass_classifiers[box_name] = dict_oneclass_classifiers[box_name]
for s in range(np.maximum(0, num_models-num_models_standard)):
    box_name = "IF" + str(s)
    oneclass_classifiers[box_name] = IsolationForest(random_state=10000*random_state+s)

# List of binary classifiers for this experiment
binary_classifiers = dict()
num_models_standard = np.minimum(4, num_models)
for s in range(num_models_standard):
    box_name = list(dict_binary_classifiers.keys())[s]
    binary_classifiers[box_name] = dict_binary_classifiers[box_name]
for s in range(np.maximum(0, num_models-num_models_standard)):
    box_name = "RF" + str(s)
    binary_classifiers[box_name] = RandomForestClassifier(random_state=10000*random_state+s)

###############
# Output file #
###############
outfile_prefix = "results/setup_greedy" + str(setup) + "/" +str(data_name) + "_n"+str(n) + "_p" + str(p) + "_a" + str(a) + "_purity" + str(purity) + "_numodels" + str(num_models) + "_seed" + str(random_state)
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
    df["Num-models"] = num_models
    df["Seed"] = random_state
    return df

###################
# Run experiments #
###################

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
