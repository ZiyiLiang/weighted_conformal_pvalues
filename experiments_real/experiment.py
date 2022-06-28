import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from scipy.io import arff, loadmat
from sklearn.model_selection import train_test_split
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
# Experiment parameters #
#########################

if True: # Input parameters
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    model_num = 1
    if len(sys.argv) != 4:
        print("Error: incorrect number of parameters.")
        quit()
    data_name = sys.argv[1]
    n = int(sys.argv[2])
    random_state = int(sys.argv[3])

else: # Default parameters
    data_name = "musk"
    n = 1000
    random_state = 2022


# Fixed experiment parameters
calib_size = 0.5
alpha_list = [0.01, 0.02, 0.05, 0.1, 0.2]
num_repetitions = 1

# List of possible one-class classifiers with desired hyper-parameters
oneclass_classifiers = {
    'SVM-rbf': OneClassSVM(kernel='rbf'),
    'SVM-sig': OneClassSVM(kernel='sigmoid'),
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

#########################
# Data-generating model #
#########################

class DataSet:

    def __init__(self, data_name, random_state=None):
        base_path = "../experiments_real/data/"
        # Load the data
        X, Y = self._load_outlier_data(base_path, data_name + ".mat")
        print("Loaded data set with {:d} samples: {:d} inliers, {:d} outliers.".format(len(Y), np.sum(Y==0), np.sum(Y==1)))

        # Extract test set
        if random_state is not None:
            np.random.seed(random_state)            
        idx_in = np.where(Y==0)[0]
        idx_out = np.where(Y==1)[0]
        idx_test_out = np.random.choice(idx_out, int(len(idx_out)/2), replace=False)
        idx_test_in = np.random.choice(idx_in, len(idx_test_out), replace=False)
        idx_test = np.append(idx_test_out, idx_test_in)
        idx_train = np.setdiff1d(np.arange(len(Y)), idx_test)
        np.random.shuffle(idx_train)
        np.random.shuffle(idx_test)
        self.X = X[idx_train]
        self.Y = Y[idx_train]
        self.X_test = X[idx_test]
        self.Y_test = Y[idx_test]
        self.n_out = np.sum(self.Y==1)

    def _load_outlier_data(self, base_path, filename):
        if filename.endswith('.csv'):
            data_raw = pd.pandas.read_csv(base_path + filename)
        elif filename.endswith('.mat'):
            if 'http' in filename:
                mat = mat73.loadmat(base_path + filename)
            else:
                mat = loadmat(base_path + filename)
            X = mat['X']
            Y = mat['y'].reshape(-1,1)
            data = np.concatenate((X,Y),axis=1)
            index   = [str(i) for i in range(0, len(Y))]
            columns = ["X_" + str(i) for i in range(0, X.shape[1])]
            columns.append("Class")
            data_raw = pd.DataFrame(data=data,index=index,columns=columns)
        elif filename.endswith('.arff'):
            data = arff.loadarff(base_path + filename)
            data_raw = pd.DataFrame(data[0])
            data_raw = data_raw.drop(columns=['id'])
            data_raw = data_raw.rename(columns={"outlier": "Class"})
            data_raw['Class'] = (data_raw['Class']==b'yes').astype(float)

        X = np.array(data_raw.drop("Class", axis=1))
        p = X.shape[1]
        Y = np.array(data_raw["Class"]).astype(int)
        return X, Y

    def sample_test(self, n=None, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)            
        if n is None:
            idx_sample = np.arange(len(self.Y_test))
        else:
            idx_sample = np.random.choice(len(self.Y_test), n)
        return self.X_test[idx_sample], self.Y_test[idx_sample]

    def sample(self, n=None, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)    
        if n is None:
            n = np.sum(self.Y==1)
        
        idx_in = np.where(self.Y==0)[0]
        idx_out = np.where(self.Y==1)[0]
        n = np.minimum(n, len(idx_out))
        idx_sample_out = np.random.choice(idx_out, n, replace=False)
        idx_sample = np.append(idx_in, idx_sample_out)
        np.random.shuffle(idx_sample)
        
        return self.X[idx_sample], self.Y[idx_sample]
    
####################################
# Reduce the sample size if needed #
####################################
dataset = DataSet(data_name, random_state=0)
n = np.minimum(n, dataset.n_out)

###############
# Output file #
###############
outfile_prefix = "results/" + str(data_name) + "_n"+str(n) + "_seed" + str(random_state)
outfile = outfile_prefix + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

if os.path.exists(outfile):
    print("Output file found. Quitting!")
    exit(0)

# Header for results file
def add_header(df):
    df["Data"] = data_name
    df["n"] = n
    df["Seed"] = random_state
    return df

###################
# Run experiments #
###################

def run_experiment(dataset, random_state):
    # Sample the training/calibration data
    X, Y = dataset.sample(n)
    X_in = X[Y==0]
    X_out = X[Y==1]
    # Sample the test data
    X_test, Y_test = dataset.sample_test()

    print("--------------------")
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

    ## Conformal p-values via weighted one-class classification
    print("Running {:d} weighted one-class classifiers...".format(len(oneclass_classifiers)))
    sys.stdout.flush()
    for occ_name in tqdm(oneclass_classifiers.keys()):
        occ = oneclass_classifiers[occ_name]
        method = IntegrativeConformal(X_in, X_out, bboxes_one=[occ], calib_size=calib_size, tuning=True, progress=False, verbose=False)
        pvals_test, pvals_test_0, pvals_test_1 = method.compute_pvalues(X_test, return_prepvals=True)
        results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
        results_tmp["Method"] = "Weighted One-Class"
        results_tmp["Model"] = occ_name
        results_tmp["E_U1_Y0"] = np.mean(pvals_test_1)
        results_tmp["1/log(n1+1)"] = 1/np.log(int(X_out.shape[0]*calib_size)+1.0)
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

    ## Conformal p-values via binary ensemble (no weighting)
    print("Running binary classifiers with learning ensemble (without weighting)...")
    sys.stdout.flush()
    bboxes_one = list(oneclass_classifiers.values())
    bboxes_two = list(binary_classifiers.values())
    method = IntegrativeConformal(X_in, X_out,
                                       bboxes_two=bboxes_two,
                                       calib_size=calib_size, ratio=False, tuning=True, progress=True, verbose=False)
    pvals_test = method.compute_pvalues(X_test)
    results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    results_tmp["Method"] = "Ensemble (binary, unweighted)"
    results_tmp["Model"] = "Ensemble"
    results_tmp["E_U1_Y0"] = np.nan
    results_tmp["1/log(n1+1)"] = np.nan
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
