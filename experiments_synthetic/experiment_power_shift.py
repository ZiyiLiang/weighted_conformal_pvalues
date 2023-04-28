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
from scipy.special import zeta

import os, sys
sys.path.append("../methods")

from models import GaussianMixture, ConcentricCircles, ConcentricCirclesMixture, BinomialModel
from methods_split import BinaryConformal, OneClassConformal, IntegrativeConformal

from util_experiments import eval_pvalues, estimate_beta_mixture, sample_beta_mixture

import decimal
ctx = decimal.Context()

# 9 digits should be enough
ctx.prec = 9

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

#########################
# Data-generating model #
#########################

class DataSet:
    def __init__(self, data_name, p, a_labeled, a_test, random_state=None):
        if data_name=="circles":
            self.model_labeled = ConcentricCircles(p, a_labeled, random_state=random_state)
            self.model_test = ConcentricCircles(p, a_test, random_state=random_state)
        elif data_name=="circles-mixed":
            self.model_labeled = ConcentricCirclesMixture(p, a_labeled, random_state=random_state)
            self.model_test = ConcentricCirclesMixture(p, a_test, random_state=random_state)
        elif data_name=="binomial":
            self.model_labeled = BinomialModel(p, a_labeled, random_state=random_state)
            self.model_test = BinomialModel(p, a_test, random_state=random_state)
        else:
            print("Error: unknown model name!")
            exit(0)

    def sample_labeled(self, n, purity):
        return self.model_labeled.sample(n, purity)

    def sample_test(self, n, purity):
        return self.model_test.sample(n, purity)

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
    a_shift = float(sys.argv[6])
    purity = float(sys.argv[7])
    gamma_model = float(sys.argv[8])
    random_state = int(sys.argv[9])

else: # Default parameters
    setup = 1
    data_name = "circles-mixed"
    n = 100
    p = 1000
    a = 0.7
    a_shift = 0
    purity = 0.5
    gamma_model = 1
    random_state = 2022

# Fixed experiment parameters
n_test = 1000
purity_test = 0.5
calib_size = 0.5
alpha_list = [0.01, 0.02, 0.05, 0.1, 0.2]
num_repetitions = 1

gamma_euler = 0.577

# Define outlier shift
# Assumes a_shift <= 1
# No shift if a_shift = 0
a_labeled = a
a_test = (1.0 - a_shift) * a

# Choose a family of one-class classifiers
bboxes_one_in = [OneClassSVM(kernel='rbf', gamma=0.0005)]
bboxes_one_out = [OneClassSVM(kernel='rbf', gamma=gamma_model)] 

###############
# Output file #
###############
outfile_prefix = "results/setup_power_shift" + str(setup) + "/" +str(data_name) + "_n"+str(n) + "_p" + str(p) + "_a" + str(a) + "_as" + str(a_shift) + "_purity" + str(purity) + "_gamma" + float_to_str(gamma_model) + "_seed" + str(random_state)
outfile = outfile_prefix + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

# Header for results file
def add_header(df):
    df["Setup"] = setup
    df["Data"] = data_name
    df["n"] = n
    df["p"] = p
    df["Signal"] = a
    df["Shift"] = a_shift
    df["Purity"] = purity
    df["Gamma"] = gamma_model
    df["Seed"] = random_state
    return df

###################
# Run experiments #
###################


def run_experiment(dataset, random_state):
    # Sample the training/calibration data
    X, Y = dataset.sample_labeled(n, purity)
    X_in = X[Y==0]
    X_out = X[Y==1]
    # Sample the test data
    X_test, Y_test = dataset.sample_test(n_test, purity_test)

    # Initialize result data frame
    results = pd.DataFrame({})

    ## Conformal p-values via learning ensemble (no weighting)
    print("Running weighted classifiers with learning ensemble (without weighting)...")
    sys.stdout.flush()
    method = IntegrativeConformal(X_in, X_out,
                                  bboxes_one=bboxes_one_in, bboxes_one_out=bboxes_one_out,
                                  calib_size=calib_size, ratio=False, tuning=True, progress=True, verbose=False)
    pvals_test = method.compute_pvalues(X_test)
    results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    results_tmp["Method"] = "Ensemble (one-class, unweighted)"
    results_tmp["Model"] = "Ensemble"
    results_tmp["E_U1_Y0"] = np.nan
    results_tmp["E_U1_Y0_approx"] = np.nan
    results_tmp["1/log(n1+1)"] = np.nan
    results_tmp["xi-2"] = np.nan
    results_tmp["xi-2-hat"] = np.nan 
    results_tmp["xi-3-hat"] = np.nan 
    results_tmp["xi"] = np.nan
    results = pd.concat([results, results_tmp])

    ## Conformal p-values via weighted one-class classification and learning ensemble
    print("Running weighted classifiers with learning ensemble...")
    sys.stdout.flush()
    method = IntegrativeConformal(X_in, X_out,
                                  bboxes_one=bboxes_one_in, bboxes_one_out=bboxes_one_out,
                                  calib_size=calib_size, tuning=True, progress=True, verbose=False)
    pvals_test, pvals_test_0, pvals_test_1 = method.compute_pvalues(X_test, return_prepvals=True)
    _, _, pvals_train_in_1 = method.compute_pvalues(method.X_in_train, return_prepvals=True)

    # Power analysis based on beta-mixture distribution
    pvals_ref = pvals_train_in_1
    nu, lamb = estimate_beta_mixture(pvals_test_1, pvals_ref)
    n1 = int(X_out.shape[0]*calib_size)
    if np.abs(nu-1) < 1e-3:
        informativeness = 1.0/(gamma_euler+np.log(n1+1.0))
    elif nu < 1:
        informativeness = 1 / (nu * zeta(2-nu) * np.power(n1+1, 1-nu))
    else:
        informativeness = (nu-1)/nu * (1 - (nu-1)* zeta(2-nu) / np.power(n1+1, nu-1))

    # Debug: check whether the fitting worked well
    if False:
        sample_mix = sample_beta_mixture(len(pvals_test_1), pvals_ref, nu, lamb)

        plt.subplot(1, 5, 1)
        plt.hist(pvals_train_in_1)
        plt.title("U_1 for train inliers.\n M1: {:.3f}. M2: {:.3f}".format(np.mean(pvals_train_in_1), np.mean(pvals_train_in_1**2)))
        plt.subplot(1, 5, 2)
        plt.hist(pvals_test_1[Y_test==0])
        plt.title("U_1 for test inliers.\n M1: {:.3f}. M2: {:.3f}".format(np.mean(pvals_test_1[Y_test==0]), np.mean(pvals_test_1[Y_test==0]**2)))
        plt.subplot(1, 5, 3)
        plt.hist(pvals_test_1[Y_test==1])
        plt.title("U_1 for test outliers.\n M1: {:.3f}. M2: {:.3f}".format(np.mean(pvals_test_1[Y_test==1]), np.mean(pvals_test_1[Y_test==1]**2)))
        plt.subplot(1, 5, 4)
        plt.hist(pvals_test_1)
        plt.title("U_1 for all test points.\n M1: {:.3f}. M2: {:.3f}".format(np.mean(pvals_test_1), np.mean(pvals_test_1**2)))
        plt.subplot(1, 5, 5)
        plt.hist(sample_mix)
        plt.title(" nu: {:.3f}, lamb: {:.3f}\n U_1 for mixture sample.\n M1: {:.3f}. M2: {:.3f}".format(nu, lamb, np.mean(sample_mix), np.mean(sample_mix**2)))
        plt.show()
        
    # Store results
    results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    results_tmp["Method"] = "Ensemble"
    results_tmp["Model"] = "Ensemble"
    results_tmp["E_U1_Y0"] = np.mean(pvals_test_1[Y_test==0])
    results_tmp["E_U1_Y0_approx"] = np.mean(pvals_ref)
    results_tmp["1/log(n1+1)"] = 1.0/(gamma_euler+np.log(n1+1.0))
    results_tmp["xi-2"] = results_tmp["1/log(n1+1)"] / results_tmp["E_U1_Y0"]
    results_tmp["xi-2-hat"] = results_tmp["1/log(n1+1)"] / results_tmp["E_U1_Y0_approx"]
    results_tmp["xi-3-hat"] = informativeness/ results_tmp["E_U1_Y0_approx"]
    results_tmp["xi"] = (1/np.mean(1/pvals_test_1[Y_test==1]))/np.mean(pvals_test_1[Y_test==0])
    results = pd.concat([results, results_tmp])


    return results

# Initialize result data frame
results = pd.DataFrame({})

for r in range(num_repetitions):
    print("\nStarting repetition {:d} of {:d}:\n".format(r+1, num_repetitions))
    sys.stdout.flush()
    # Change random seed for this repetition
    random_state_new = 10*num_repetitions*random_state + r
    dataset = DataSet(data_name, p, a_labeled, a_test, random_state=random_state_new)
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
