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
calib_size = 0.5
n_test = 1000
alpha_list = [0.01, 0.02, 0.05, 0.1, 0.2]
num_repetitions = 1

# List of possible one-class classifiers with desired hyper-parameters
oneclass_classifiers = {
    'SVM-rbf': OneClassSVM(kernel='rbf'),
#    'SVM-sig': OneClassSVM(kernel='sigmoid'),
#    'SVM-pol': OneClassSVM(kernel='poly', degree=3),
#    'IF': IsolationForest(random_state=random_state),
#    'LOF': LocalOutlierFactor(novelty=True)
}

# Define list of possible two-class classifiers with desired hyper-parameters
binary_classifiers = {
#    'RF': RandomForestClassifier(random_state=random_state),
#    'KNN': KNeighborsClassifier(),
#    'SVC': SVC(probability=True),
#    'NB' : GaussianNB(),
#    'QDA': QuadraticDiscriminantAnalysis(),
    'MLP': MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,100,), random_state=random_state)
}

#########################
# Data-generating model #
#########################

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    #cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    #if negatives:
    #    cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    #else:
    #    cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels

class DataSet:

    def __init__(self, data_name, random_state=None):
        base_path = "data/"
        # Load the data
        if data_name=="digits":
            dataraw = load_digits()
            X = dataraw['data']
            Y = dataraw['target'] == 3
        elif data_name=="covtype":
            dataraw = fetch_covtype()
            X = dataraw['data']
            Y = dataraw['target'] == 1
        elif data_name=="toxicity":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=";", header=None)
            Y = (np.array(data_raw.iloc[:,-1])=='positive').astype(int)
            X = np.array(data_raw.iloc[:,:-1])
        elif data_name=="ad":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None, na_values=['?','     ?','   ?'])
            data_raw = data_raw.fillna(data_raw.median())
            Y = (np.array(data_raw.iloc[:,-1])=='ad.').astype(int)
            X = np.array(data_raw.iloc[:,:-1])
        elif data_name=="androgen":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=";", header=None)
            Y = (np.array(data_raw.iloc[:,-1])=='positive').astype(int)
            X = np.array(data_raw.iloc[:,:-1])
        elif data_name=="rejafada":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None).iloc[:,1:]
            Y = (np.array(data_raw.iloc[:,0])=='M').astype(int)
            X = np.array(data_raw.iloc[:,1:])
        elif data_name=="hepatitis":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None, na_values=['?','     ?','   ?'])
            data_raw = data_raw.fillna(data_raw.median())
            Y = (np.array(data_raw.iloc[:,0])==1).astype(int)
            X = np.array(data_raw.iloc[:,1:])
        elif data_name=="ctg":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None, na_values=['?','     ?','   ?'])
            data_raw = data_raw.fillna(data_raw.median())
            Y = (np.array(data_raw.iloc[:,-1])==2).astype(int)
            X = np.array(data_raw.iloc[:,:-1])
        elif data_name=="creditcard":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",")
            data_raw = data_raw.fillna(data_raw.median())
            Y = (np.array(data_raw.iloc[:,-1])==1).astype(int)
            X = np.array(data_raw.iloc[:,:-1])
        elif data_name=="seizures":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",")
            idx_keep = np.where(data_raw.iloc[:,-1]!=1)[0]
            Y = (np.array(data_raw.iloc[idx_keep,-1])==5).astype(int)
            X = np.array(data_raw.iloc[idx_keep,1:-1])
        elif data_name=="splice":
            data_raw = pd.pandas.read_csv(base_path + "splice.data", header=None, sep=",")
            X_raw = data_raw.iloc[:,-1]

            label_encoder = LabelEncoder()
            label_encoder.fit(np.array(['a','c','g','t','n']))

            def string_to_array(seq_string):
                seq_string = seq_string.lower().strip()
                seq_string = re.sub('[^acgt]', 'n', seq_string)
                seq_string = np.array(list(seq_string))
                return seq_string

            int_encoded = label_encoder.transform(string_to_array('acgtn'))
            onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
            int_encoded = int_encoded.reshape(len(int_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(int_encoded)

            def one_hot_encoder(seq_string):
                int_encoded = label_encoder.transform(seq_string)
                int_encoded = int_encoded.reshape(len(int_encoded), 1)
                onehot_encoded = onehot_encoder.transform(int_encoded)
                onehot_encoded = np.delete(onehot_encoded, -1, 1)
                return onehot_encoded

            X = [ one_hot_encoder(string_to_array(x)).flatten() for x in X_raw]
            X = np.stack(X)
            Y = np.array(data_raw.iloc[:,0]=="IE").astype(int)

        elif data_name=="cifar-100":
            def unpickle(file):
                import pickle
                with open(file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                return dict

            metadata_path = base_path + 'cifar-100/meta' # change this path`\
            metadata = unpickle(metadata_path)
            superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))

            data_pre_path = base_path + 'cifar-100/' # change this path
            # File paths
            data_train_path = data_pre_path + 'train'
            data_test_path = data_pre_path + 'test'
            # Read dictionary
            data_train_dict = unpickle(data_train_path)
            data_test_dict = unpickle(data_test_path)
            # Get data (change the coarse_labels if you want to use the 100 classes)
            X = data_train_dict[b'data']
            Y = np.array(data_train_dict[b'coarse_labels'])
            idx_keep = np.where((Y==18)+(Y==19)+(Y==5)+(Y==6)+(Y==14)>0)[0]
            X = X[idx_keep]
            Y = Y[idx_keep]
            X = X.astype(float)
            Y = (Y==14).astype(int)
        
        elif data_name=="cifar-10":
            cifar_10_dir = base_path + "cifar-10"
            X, _, Y, test_data, _, _ = load_cifar_10_data(cifar_10_dir)
            idx_keep = np.where((Y==0)+(Y==3)+(Y==4)+(Y==5)+(Y==6)+(Y==7)>0)[0]
            X = X[idx_keep]
            Y = (Y[idx_keep] == 0).astype(int)
            
        else:
            X, Y = self._load_outlier_data(base_path, data_name + ".mat")
        print("Loaded data set with {:d} samples: {:d} inliers, {:d} outliers.".format(len(Y), np.sum(Y==0), np.sum(Y==1)))

        # Extract test set
        if random_state is not None:
            np.random.seed(random_state)            
        idx_in = np.where(Y==0)[0]
        idx_out = np.where(Y==1)[0]
        idx_test_out = np.random.choice(idx_out, int(len(idx_out)/5), replace=False)
        idx_test_in = np.random.choice(idx_in, len(idx_test_out), replace=False)
        idx_test = np.append(idx_test_out, idx_test_in)
        idx_train = np.setdiff1d(np.arange(len(Y)), idx_test)
        np.random.shuffle(idx_train)
        np.random.shuffle(idx_test)
        self.X = X[idx_train]
        self.Y = Y[idx_train]
        self.X_test = X[idx_test]
        self.Y_test = Y[idx_test]
        self.n_in = np.sum(self.Y==0)
        self.n_out = np.sum(self.Y==1)

    def _load_outlier_data(self, base_path, filename, sep=","):
        if filename.endswith('.csv'):
            data_raw = pd.pandas.read_csv(base_path + filename, sep=sep)
            pdb.set_trace()
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
            n = np.minimum(n, len(self.Y_test))
            idx_sample = np.random.choice(len(self.Y_test), n)
        return self.X_test[idx_sample], self.Y_test[idx_sample]

    def sample(self, n_in=None, n_out=None, mislabeled_prop=0, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)    
        if n_out is None:
            n_out = np.sum(self.Y==1)
        if n_in is None:
            n_in = np.sum(self.Y==0)
        
        idx_in = np.where(self.Y==0)[0]
        idx_out = np.where(self.Y==1)[0]
        n_out = np.minimum(n_out, len(idx_out))
        n_in = np.minimum(n_in, len(idx_in))

        idx_sample_out = np.random.choice(idx_out, n_out, replace=False)
        idx_sample_in = np.random.choice(idx_in, n_in, replace=False)
        Y = self.Y
        
        if mislabeled_prop>0:
            n_out_mis = np.minimum(n_in, int(n_out*mislabeled_prop))
            idx_sample_in, idx_sample_in_out = train_test_split(idx_sample_in, test_size=n_out_mis, random_state=random_state)
            idx_sample_out = np.append(idx_sample_out, idx_sample_in_out)
            Y[idx_sample_in_out] = 1
        
        idx_sample = np.append(idx_sample_in, idx_sample_out)
        np.random.shuffle(idx_sample)
        
        return self.X[idx_sample], Y[idx_sample]
    
####################################
# Reduce the sample size if needed #
####################################
dataset = DataSet(data_name, random_state=0)
n_in = np.minimum(n_in, dataset.n_in)
n_out = np.minimum(n_out, dataset.n_out)

###############
# Output file #
###############
outfile_prefix = "results/" + str(data_name) + "_nin"+str(n_in) + "_nout"+str(n_out) + "_seed" + str(random_state)
outfile = outfile_prefix + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

if os.path.exists(outfile):
    print("Output file found. Quitting!")
    exit(0)

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
    # Sample the training/calibration data
    X, Y = dataset.sample(n_in=n_in, n_out=n_out)
    X_in = X[Y==0]
    X_out = X[Y==1]
    # Sample the test data
    X_test, Y_test = dataset.sample_test(n=n_test)

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

    # ## Conformal p-values via weighted one-class classification
    # print("Running {:d} weighted one-class classifiers...".format(len(oneclass_classifiers)))
    # sys.stdout.flush()
    # for occ_name in tqdm(oneclass_classifiers.keys()):
    #     occ = oneclass_classifiers[occ_name]
    #     method = IntegrativeConformal(X_in, X_out, bboxes_one=[occ], calib_size=calib_size, tuning=True, progress=False, verbose=False)
    #     pvals_test, pvals_test_0, pvals_test_1 = method.compute_pvalues(X_test, return_prepvals=True)
    #     results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    #     results_tmp["Method"] = "Weighted One-Class"
    #     results_tmp["Model"] = occ_name
    #     results_tmp["E_U1_Y0"] = np.mean(pvals_test_1)
    #     results_tmp["1/log(n1+1)"] = 1/np.log(int(X_out.shape[0]*calib_size)+1.0)
    #     results = pd.concat([results, results_tmp])

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

    # ## Conformal p-values via learning ensemble (no weighting)
    # print("Running weighted classifiers with learning ensemble (without weighting)...")
    # sys.stdout.flush()
    # bboxes_one = list(oneclass_classifiers.values())
    # bboxes_two = list(binary_classifiers.values())
    # method = IntegrativeConformal(X_in, X_out,
    #                                    bboxes_one=bboxes_one, bboxes_two=bboxes_two,
    #                                    calib_size=calib_size, ratio=False, tuning=True, progress=True, verbose=False)
    # pvals_test = method.compute_pvalues(X_test)
    # results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    # results_tmp["Method"] = "Ensemble (mixed, unweighted)"
    # results_tmp["Model"] = "Ensemble"
    # results_tmp["E_U1_Y0"] = np.nan
    # results_tmp["1/log(n1+1)"] = np.nan
    # results = pd.concat([results, results_tmp])

    # ## Conformal p-values via learning ensemble (one-class, no weighting)
    # print("Running weighted classifiers with learning ensemble (one-class, without weighting)...")
    # sys.stdout.flush()
    # bboxes_one = list(oneclass_classifiers.values())
    # bboxes_two = list(binary_classifiers.values())
    # method = IntegrativeConformal(X_in, X_out,
    #                                    bboxes_one=bboxes_one,
    #                                    calib_size=calib_size, ratio=False, tuning=True, progress=True, verbose=False)
    # pvals_test = method.compute_pvalues(X_test)
    # results_tmp = eval_pvalues(pvals_test, Y_test, alpha_list)
    # results_tmp["Method"] = "Ensemble (one-class, unweighted)"
    # results_tmp["Model"] = "Ensemble"
    # results_tmp["E_U1_Y0"] = np.nan
    # results_tmp["1/log(n1+1)"] = np.nan
    # results = pd.concat([results, results_tmp])

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
