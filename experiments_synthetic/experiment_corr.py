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
    data_name = "circles-mixed"
    n = 100
    p = 1000
    a = 0.7
    purity = 0.5
    random_state = 2022

# Fixed experiment parameters
n_test = 2
purity_test = 1
calib_size = 0.5
num_repetitions = 100

# Choose a family of one-class classifiers
bbox_occ_list = {'SVM-rbf':OneClassSVM(kernel='rbf', degree=3),
                 'SVM-sig':OneClassSVM(kernel='sigmoid', degree=3),
                 'SVM-poly':OneClassSVM(kernel='poly', degree=3)
                }

###############
# Output file #
###############
outfile_prefix = "results/setup_corr" + str(setup) + "/" +str(data_name) + "_n"+str(n) + "_p" + str(p) + "_a" + str(a) + "_purity" + str(purity) + "_seed" + str(random_state)
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

###################
# Run experiments #
###################

def measure_correlation(n, random_state):
    dataset = DataSet(data_name, random_state=random_state)
    X, Y = dataset.sample(n, purity)
    X_test, Y_test = dataset.sample(n_test, 1)

    # Extract the inliers from the data
    X_in = X[Y==0]
    X_out = X[Y==1]

    # Compute the OCC p-values
    method_occ = OneClassConformal(X_in, bbox_occ_list['SVM-rbf'], calib_size=0.5, verbose=False)
    pvals_occ = method_occ.compute_pvalues(X_test)
    
    # Compute the integrative p-values
    method_int = IntegrativeConformal(X_in, X_out, 
                                      bboxes_one=list(bbox_occ_list.values()), 
                                      bboxes_one_out=list(bbox_occ_list.values()),
                                      calib_size=0.5, verbose=False, progress=False, tuning=True)
    pvals_int = method_int.compute_pvalues(X_test)
    return pvals_int, pvals_occ


# Run experiment
P_vals_int = np.zeros((num_repetitions, n_test))
P_vals_occ = np.zeros((num_repetitions, n_test))
for r in tqdm(range(num_repetitions)):
    P_vals_int[r], P_vals_occ[r] = measure_correlation(n, random_state*1e6+r)
P_cov_int = np.corrcoef(P_vals_int.T)
P_cov_occ = np.corrcoef(P_vals_occ.T)
corr_int = P_cov_int[0,1]
corr_occ = P_cov_occ[0,1]
corr_theory = 1.0/(purity*0.5*n+2)

results = pd.DataFrame({'Integrative':[corr_int], 'OCC':[corr_occ], 'OCC-Theory':[corr_theory]})
results = add_header(results)

# Save results
results.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()
