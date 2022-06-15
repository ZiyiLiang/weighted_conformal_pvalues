import numpy as np
import pandas as pd
import seaborn as sns
import statistics as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb


import sys
import os

# if os.path.isdir('C:/Users/liang'):
#     local_machine = 1
#     print('running experiments on local machine')
#     sys.stdout.flush()
# else:
#     local_machine = 0
#     print('running experiments on virtual machine')
#     sys.stdout.flush()


# if local_machine:
#     base_path = "C:/Users/liang/OneDrive/Desktop/CLRA/codes/realdata/data/"
# else:
#     base_path = ''

#if os.path.isdir('/mnt/c/users/liang'):
if os.path.isdir('/home1/ziyilian'):
    local_machine = 0
    print('running experiments on virtual machine')
    sys.stdout.flush()
else:
    local_machine = 1
    print('running experiments on linux machine')
    sys.stdout.flush()

if local_machine:
    base_path = "/mnt/c/users/liang/OneDrive/Desktop/CLRA/codes/realdata/data/"
else:
    base_path = '/home1/ziyilian/CLRA/realdata/data/'

# sys.path.append('C:/Users/liang/OneDrive/Desktop/CLRA/codes/realdata/code')
# sys.path.append('C:/Users/liang/OneDrive/Desktop/CLRA/codes/realdata/data')
# sys.path.append('C:/Users/liang/OneDrive/Desktop/CLRA/related_resources/arc')
# sys.path.append('C:/Users/liang/OneDrive/Desktop/CLRA/related_resources/cqr-comparison')
# sys.path.append('C:/Users/liang/OneDrive/Desktop/CLRA/related_resources/cqr')
# sys.path.append('C:/Users/liang/OneDrive/Desktop/CLRA/related_resources/conditional-conformal-pvalues')
# sys.path.append('/mnt/c/users/liang/OneDrive/Desktop/CLRA/codes/realdata/code')
# sys.path.append('/mnt/c/users/liang/OneDrive/Desktop/CLRA/codes/realdata/data')
# sys.path.append('/mnt/c/users/liang/OneDrive/Desktop/CLRA/related_resources/arc')
# sys.path.append('/mnt/c/users/liang/OneDrive/Desktop/CLRA/related_resources/cqr-comparison')
# sys.path.append('/mnt/c/users/liang/OneDrive/Desktop/CLRA/related_resources/cqr')
# sys.path.append('/mnt/c/users/liang/OneDrive/Desktop/CLRA/related_resources/conditional-conformal-pvalues')
sys.path.append('/home1/ziyilian/CLRA/realdata/code')
sys.path.append('/home1/ziyilian/CLRA/realdata/data')
sys.path.append('/home1/ziyilian/CLRA/arc')
sys.path.append('/home1/ziyilian/CLRA/cqr-comparison')
sys.path.append('/home1/ziyilian/CLRA/cqr')
sys.path.append('/home1/ziyilian/CLRA/conditional-conformal-pvalues')

#########################
# Experiment parameters #
#########################
from sklearn.base import clone
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from outlier_dataset import get_data_is_outlier
from methods_experiment import oc, bc, clra


num_train = 10
num_test = 10
purity = 0.9
alpha = 0.1

# Parse input arguments
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
print("====")
print(len(sys.argv))
if len(sys.argv) != 3:
    print(sys.argv)
    print("Error: incorrect number of parameters.")
    quit()

data_set_id = int(sys.argv[1])
random_state = int(sys.argv[2])

DATASET_LIST =  ["cover.mat",
                 "creditcard.csv",
                 "shuttle.mat",
                 "mammography.mat",
                 "pendigits.mat",
                 "ALOI_withoutdupl.arff",
                 "http.mat"
                 ]


bc_methods = {
    'BC': bc,
    }
oc_methods = {
    'OCC': oc,
    'CLRA': clra, 
    }
bc_bb = {
    'RFC': RandomForestClassifier(),
    'SVC': SVC(kernel='rbf', C=1, probability=True),
    'MLP': MLPClassifier(),
    'k-NN': KNeighborsClassifier(n_neighbors=3),
    'QDA': QuadraticDiscriminantAnalysis()
}
contamination = 0.1
oc_bb = {
    'SVM': svm.OneClassSVM(nu=contamination, kernel="rbf"),
    'ISF': IsolationForest(contamination=contamination, behaviour="new"),
    'LOF': LocalOutlierFactor(novelty=True, contamination=contamination)
}

dataset_name = DATASET_LIST[data_set_id]
print("Dataset:\n  " + dataset_name)
sys.stdout.flush()

dataset, is_outlier = get_data_is_outlier(dataset_name, base_path)

n_outliers = int(sum(is_outlier))
tot_n = int(len(is_outlier))
n_clean = int(tot_n - n_outliers)
n_train = [int(np.floor(n_clean*0.5)), int(np.floor(n_outliers*0.5))]
n_calib = [min(2000, np.round(0.5*n_train[0]).astype(int)),
           min(2000, np.round(0.5*n_train[1]).astype(int))]
# set a smaller test size for pendigits dataset since not enough outliers 
n_test = 700 if data_set_id == 4 else 1000
purity_test = 0.9

print("n_outliers {:d}, tot_n {:d}, n_clean {:d}, n_train_in {:d},n_train_out {:d},\
       n_calib_in {:d}, n_calib_out {:d}, n_test {:d},".format(n_outliers, tot_n, \
       n_clean, n_train[0], n_train[1], n_calib[0], n_calib[1], n_test))
sys.stdout.flush()



###################
# Output location #
###################
#out_dir = "C:/Users/liang/OneDrive/Desktop/CLRA/codes/realdata/results/"
#out_dir = "/mnt/c/users/liang/OneDrive/Desktop/CLRA/codes/realdata/results/"
out_dir = "/home1/ziyilian/CLRA/realdata/results/"
out_file = "dataset_" + dataset_name + "_"
out_file += "seed_" + str(random_state) + ".csv"
print("Output directory for this experiment:\n  " + out_dir)
print("Output file for this experiment:\n  " + out_file)
out_file = out_dir + out_file



################
# Data manager #
################

class DataManager:
    def __init__(self, data, is_outlier):
        self.data_clean = data[is_outlier==0]
        self.data_outlier = data[is_outlier==1]
        self.X_test_clean = None
        self.Y_test_clean = None
        self.X_test_outlier = None
        self.Y_test_outlier = None
        assert self.data_clean.shape[1] == self.data_outlier.shape[1], 'number of feature is different for two classes'
        self.p = self.data_clean.shape[1]

    def sample_train(self, n, random_state=2022):
        # n should be a tuple specifying training size for both classes
        # Divide full data set into training and test parts
        X_train_clean, self.X_test_clean, Y_train_clean, self.Y_test_clean =\
        train_test_split(self.data_clean, is_outlier[is_outlier==0], train_size=n[0], random_state=random_state)
        X_train_outlier, self.X_test_outlier, Y_train_outlier, self.Y_test_outlier =\
        train_test_split(self.data_outlier, is_outlier[is_outlier==1], train_size=n[1], random_state=random_state)
        X_train = np.append(X_train_clean, X_train_outlier, axis=0)
        Y_train = np.append(Y_train_clean, Y_train_outlier)
        return X_train, Y_train 

    def sample_fit_calib(self, n_train, n_cal, random_state=2022):
        # n_cal should be a tuple specifying calibration size for both classes
        X_train, Y_train = self.sample_train(n_train, random_state=random_state)
        X_fit, Y_fit = [[]]*2, [[]]*2
        X_cal, Y_cal = [[]]*2, [[]]*2
        for i in range(2):
            X_fit[i], X_cal[i], Y_fit[i], Y_cal[i] =\
            train_test_split(X_train[np.where(Y_train==i)[0]], Y_train[np.where(Y_train==i)[0]], 
                            test_size=n_cal[i], random_state = random_state)
        return X_fit, Y_fit, X_cal, Y_cal

    def sample_test(self, n_test, purity_test=0.5, random_state=2022):
        if (self.X_test_clean is None):
            print("Error: must call sample_train method first!")
            return None

        n_clean = np.round(n_test*purity_test).astype(int)
        n_outlier = n_test - n_clean
        assert n_outlier < len(self.Y_test_outlier), 'Not enough outliers for the test set.'

        np.random.seed(random_state)
        # Select clean samples
        idx = np.random.choice(np.arange(len(self.Y_test_clean)), n_clean, replace=False)
        
        X_clean = self.X_test_clean[idx]
        Y_clean = self.Y_test_clean[idx]
        # Select outlier samples
        idx = np.random.choice(np.arange(len(self.Y_test_outlier)), n_outlier, replace=False)
        X_outlier = self.X_test_outlier[idx]
        Y_outlier = self.Y_test_outlier[idx]
        # Mix clean and outlier samples
        X_test = np.append(X_clean, X_outlier, axis=0)
        Y_test = np.append(Y_clean, Y_outlier)
        
        return X_test, Y_test


#####################
# Define experiment #
#####################
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

def run_experiment(data_manager, bc_methods, oc_methods, bc_bb, oc_bb, num_test,
                   purity_test, results, alpha=0.1, random_state=2021):
  
    X_fit, Y_fit, X_cal, Y_cal = data_manager.sample_fit_calib(n_train, n_calib, random_state=random_state)

    # train the bc black boxes 
    # add the outlier calibration data to the training data
    X_train_bc = np.concatenate((X_fit[0], X_fit[1], X_cal[1]), axis=0)
    Y_train_bc = np.concatenate((Y_fit[0], Y_fit[1], Y_cal[1]))
    # create a copy of the bb, fit the copy for this train iteration
    bc_bb_new = bc_bb.copy()
    for box_name in bc_bb:
        bc_bb_new[box_name].fit(X_train_bc, Y_train_bc)

    oc_bb_new = oc_bb.copy()
    # train the oc black boxes
    for box_name in oc_bb:
        classifiers = [[]]*2
        for i in range(2):
            classifiers[i] = clone(oc_bb[box_name])
            classifiers[i].fit(X_fit[i])
        # update the dictionary with the trained bb tuple
        oc_bb_new[box_name] = classifiers
    
    # Evaluate the performance on test data
    for test_idx in tqdm(range(num_test)):
        # sample the test set
        random_state_new = random_state + 10000 * test_idx
        X_test, Y_test = data_manager.sample_test(n_test, purity_test=purity_test, random_state=random_state_new)
        #lambda_par = 0.5
        for bc_name in bc_methods:
            for box_name in bc_bb_new:
                black_box = bc_bb_new[box_name]
                # only pass the null calibration data to the function
                X_cal_new, Y_cal_new = X_cal[0], Y_cal[0]
                # Apply outlier detection method
                pvals = bc_methods[bc_name](X_cal_new, Y_cal_new, X_test, Y_test, black_box)

                # Apply SBH
                reject, _, _, _ = multipletests(pvals, alpha=alpha/purity_test, method='fdr_bh')

                # Evaluate FDP and Power
                rejections = np.sum(reject)
                if rejections > 0:
                    fdp = np.sum(reject[np.where(Y_test==0)[0]])/reject.shape[0] 
                    power = np.sum(reject[np.where(Y_test==1)[0]])/np.sum(Y_test)
                else:
                    fdp = 0
                    power = 0

                res_tmp = {'Method':bc_name, 'Black Box': box_name, 'Test_idx': test_idx, 'Train_idx': random_state,
                            'Alpha':alpha, 'Rejections':rejections, 'FDR':fdp, 'Power':power}
                res_tmp = pd.DataFrame(res_tmp, index=[0])
                results = pd.concat([results, res_tmp])

        for oc_name in oc_methods:
            for box_name in oc_bb_new:
                black_box = oc_bb_new[box_name]

                if (oc_name == 'OCC'):
                    # only pass the null calibration data to the function
                    X_cal_new, Y_cal_new = X_cal[0], Y_cal[0]
                    # only need bb for inliers
                    black_box_new = black_box[0]
                    pvals = oc_methods[oc_name](X_cal_new, Y_cal_new, X_test, Y_test, black_box_new)
                else: 
                    # Apply outlier detection method
                    pvals = oc_methods[oc_name](X_cal, Y_cal, X_test, Y_test, black_box)       

                # Apply SBH
                reject, _, _, _ = multipletests(pvals, alpha=alpha/purity_test, method='fdr_bh')
                
                # Evaluate FDP and Power     
                rejections = np.sum(reject)
                if rejections > 0:
                    fdp = np.sum(reject[np.where(Y_test==0)[0]])/reject.shape[0] 
                    power = np.sum(reject[np.where(Y_test==1)[0]])/np.sum(Y_test)
                else:
                    fdp = 0
                    power = 0

                res_tmp = {'Method':oc_name, 'Black Box': box_name, 'Test_idx': test_idx, 'Train_idx': random_state,
                            'Alpha':alpha, 'Rejections':rejections, 'FDR':fdp, 'Power':power}
                res_tmp = pd.DataFrame(res_tmp, index=[0])
                results = pd.concat([results, res_tmp])

    return results



#######################
# Run all experiments #
#######################

# Initialize data manager with real data set
data_manager = DataManager(dataset, is_outlier)

# Run all the experiments
results = pd.DataFrame({})
for train_index in tqdm(range(num_train)):
    print("train index {:d}".format(train_index))
    sys.stdout.flush()
    random_state_new = (random_state-1)*num_train + train_index   # the training index out of all experiments
    results = run_experiment(data_manager, bc_methods, oc_methods, bc_bb, oc_bb, num_test,
                   purity_test, results, alpha=0.1, random_state=random_state_new)

# Save results on file
if out_file is not None:
    print("Saving file with {:d} rows".format(results.shape[0]))
    results.to_csv(out_file)


print("Output file for this experiment:\n  " + out_file)  
