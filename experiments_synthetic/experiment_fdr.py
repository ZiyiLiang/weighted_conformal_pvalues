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
from methods_fdr import IntegrativeConformalFDR

from util_experiments import eval_pvalues

#########################
# Experiment parameters #
#########################

if True: # Input parameters
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    model_num = 1
    if len(sys.argv) != 10:
        print("Error: incorrect number of parameters.")
        quit()

    setup = int(sys.argv[1])
    data_name = sys.argv[2]
    n = int(sys.argv[3])
    p = int(sys.argv[4])
    a = float(sys.argv[5])
    purity = float(sys.argv[6])
    alpha = float(sys.argv[7])
    n_test = int(sys.argv[8])
    random_state = int(sys.argv[9])

else: # Default parameters
    setup = 1
    data_name = "circles-mixed"
    n = 1000
    p = 1000
    a = 0.4
    purity = 0.5
    alpha = 0.1
    n_test = 10
    random_state = 2022


# Fixed experiment parameters
purity_test = 0.5
calib_size = 0.5
num_repetitions = 2
J_max = 10

# List of possible one-class classifiers with desired hyper-parameters
oneclass_classifiers = {
    'SVM-rbf': OneClassSVM(kernel='rbf', degree=3),
    'SVM-sig': OneClassSVM(kernel='sigmoid', degree=3),
    'SVM-pol': OneClassSVM(kernel='poly', degree=3),
    'IF': IsolationForest(random_state=random_state),
    'LOF': LocalOutlierFactor(novelty=True)
}

# Define list of possible two-class classifiers with desired hyper-parameters
binary_classifiers = {
    'RF': RandomForestClassifier(random_state=random_state),
    'KNN': KNeighborsClassifier(),
    'SVC': SVC(probability=True),
    'NB' : GaussianNB(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'MLP': MLPClassifier(max_iter=500, random_state=random_state)
}

###############
# Output file #
###############
outfile_prefix = "results/setup_fdr" + str(setup) + "/" +str(data_name) + "_n"+str(n) + "_p" + str(p) + "_a" + str(a) + "_purity" + str(purity) + "_alpha"+str(alpha) + "_ntest" + str(n_test) + "_seed" + str(random_state)
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
    df["Alpha"] = alpha
    df["Seed"] = random_state
    df["n_test"] = n_test
    return df

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

    def sample(self, n, purity, random_state=None):
        return self.model.sample(n, purity, random_state=random_state)

###################
# Run experiments #
###################

def eval_discoveries(reject, Y):
    is_nonnull = (Y==1)
    rejections = np.sum(reject)
    if rejections>0:
        fdp = 1-np.mean(is_nonnull[np.where(reject)[0]])
        power = np.sum(is_nonnull[np.where(reject)[0]]) / np.sum(is_nonnull)
    else:
        fdp = 0
        power = 0
    return fdp, power


def run_experiment(dataset, random_state):
    # Sample the training/calibration data
    X, Y = dataset.sample(n, purity)
    X_in = X[Y==0]
    X_out = X[Y==1]
    # Sample the test data
    X_test, Y_test = dataset.sample(n_test, purity_test)

    # Initialize result data frame
    results = pd.DataFrame({})

    ## Conformal p-values via weighted one-class classification and learning ensemble
    print("Running weighted classifiers with learning ensemble...")
    sys.stdout.flush()
    bboxes_one = list(oneclass_classifiers.values())
    bboxes_two = list(binary_classifiers.values())
    method = IntegrativeConformal(X_in, X_out,
                                  bboxes_one=bboxes_one, bboxes_two=bboxes_two,
                                  calib_size=calib_size, tuning=True, progress=True, verbose=False)
    pvals_test, pvals_test_0, pvals_test_1 = method.compute_pvalues(X_test, return_prepvals=True)

    icfdr = IntegrativeConformalFDR(method)

    # Apply the new method with no loo
    reject_sel, pruned_sel = icfdr.filter_fdr_conditional(X_test, alpha, J_max=J_max, loo='none')
    fdp_sel, power_sel = eval_discoveries(reject_sel, Y_test)    
    results_tmp = pd.DataFrame({"Method":["Selective"], "FDP":[fdp_sel], "Power":[power_sel], "LOO":['none'], "Pruned":[pruned_sel]})
    results = pd.concat([results, results_tmp])

    # Apply the new method with 'median' loo
    reject_sel, pruned_sel = icfdr.filter_fdr_conditional(X_test, alpha, J_max=J_max, loo='median')
    fdp_sel, power_sel = eval_discoveries(reject_sel, Y_test)    
    results_tmp = pd.DataFrame({"Method":["Selective"], "FDP":[fdp_sel], "Power":[power_sel], "LOO":['median'], "Pruned":[pruned_sel]})
    results = pd.concat([results, results_tmp])

    # Apply the new method with 'min' loo
    reject_sel, pruned_sel = icfdr.filter_fdr_conditional(X_test, alpha, J_max=J_max, loo='min')
    fdp_sel, power_sel = eval_discoveries(reject_sel, Y_test)    
    results_tmp = pd.DataFrame({"Method":["Selective"], "FDP":[fdp_sel], "Power":[power_sel], "LOO":['min'], "Pruned":[pruned_sel]})
    results = pd.concat([results, results_tmp])

    # Apply the regular BH
    reject_bh = icfdr.filter_fdr_bh(X_test, alpha)
    fdp_bh, power_bh = eval_discoveries(reject_bh, Y_test)
    results_tmp = pd.DataFrame({"Method":["BH"], "FDP":[fdp_bh], "Power":[power_bh], "LOO":["none"], "Pruned":[False]})
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
