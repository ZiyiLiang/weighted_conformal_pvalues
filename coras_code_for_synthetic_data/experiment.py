import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from sklearn.base import clone
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
from methods import oc, bc, clr, clra, clra_cvpp
from conditional_calib_methods import cBH, cBH_med, clra_cvpp_cBH, clra_cvpp_cBH_loo
from toy_models import Model_CC, Model_Gaussian, Model_Binomial

## define experiment parameters
MODEL_LIST = ["Circles", "Gaussian", "Binomial"]

bc_methods = {
    'Binary': bc,
    }
oc_methods = {
    'One-class': oc,
    'CLRA': clra, 
    #'CLR': clr,
    #'cBH': cBH,
    #'cBH_med': cBH_med,
    'clra_cvpp': clra_cvpp,
    #'clra_cvpp_cBH': clra_cvpp_cBH,
    #'clra_cvpp_cBH_loo': clra_cvpp_cBH_loo,
    }
bc_bb = {
    #'RFC': RandomForestClassifier(),
    'SVC': SVC(kernel='rbf', C=1, probability=True),
    #'MLP': MLPClassifier(),
    #'k-NN': KNeighborsClassifier(n_neighbors=3),
    #'QDA': QuadraticDiscriminantAnalysis()
}
oc_bb = {
    'SVM': svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.1),
    #'ISF': IsolationForest(contamination = 'auto'),
    #'LOF': LocalOutlierFactor(novelty=True)
}

model_id = int(sys.argv[1])
random_state = int(sys.argv[2])
model_name = MODEL_LIST[model_id]

###################
# Output location #
###################
out_dir = "/mnt/c/users/liang/OneDrive/Desktop/CLRA/synthetic_data/results/"

out_file = "model_" + model_name + "_"
out_file += "seed_" + str(random_state) + ".csv"
print("Output directory for this experiment:\n  " + out_dir)
print("Output file for this experiment:\n  " + out_file)
out_file = out_dir + out_file

## define a single experiment
def run_experiment(data_model, purity, bc_methods, oc_methods, bc_bb, oc_bb, alpha=0.1, experiment=0,random_state=2021):
  # Generate the training/calibration data
  n_data = 2000               # Number of observations
  purity_data = purity        # Proportion of inliers
  X_data, Y_data = model.sample(n_data, purity=purity_data, random_state=random_state+1)
  # idx_inliers = np.where(Y_data==0)
  # idx_outliers = np.where(Y_data==1)
  # Y_data[idx_inliers]=1
  # Y_data[idx_outliers]=0

  n_test = 1000            # Number of observations
  purity_test = 0.5         # Proportion of inliers
  X_test, Y_test = model.sample(n_test, purity=purity_test, random_state=random_state+2)
  # idx_inliers = np.where(Y_test==0)
  # idx_outliers = np.where(Y_test==1)
  # Y_test[idx_inliers]=1
  # Y_test[idx_outliers]=0

  #pdb.set_trace()

  results = pd.DataFrame()
  for bc_name in bc_methods:
    for box_name in bc_bb:
      black_box = bc_bb[box_name]
      # Apply outlier detection method
      pvals = bc_methods[bc_name](X_data, Y_data, X_test, Y_test, black_box)

      # Apply BH
      reject, _, _, _ = multipletests(pvals, alpha=alpha/purity_test, method='fdr_bh')

      # Evaluate FDP and Power
      rejections = np.sum(reject)
      if rejections > 0:
        fdp = np.sum(reject[np.where(Y_test==0)[0]])/reject.shape[0] 
        power = np.sum(reject[np.where(Y_test==1)[0]])/np.sum(Y_test)
      else:
        fdp = 0
        power = 0

      res_tmp = {'Method':bc_name, 'Black Box': box_name, 'Experiment': experiment, 'Dimension': p,
                 'Purity': purity, 'Alpha':alpha, 'Rejections':rejections, 'FDR':fdp, 'Power':power}
      res_tmp = pd.DataFrame(res_tmp, index=[0])
      results = pd.concat([results, res_tmp])

  for oc_name in oc_methods:
    for box_name in oc_bb:
      black_box = oc_bb[box_name]

      if ('cBH' in oc_name):
        reject = oc_methods[oc_name](X_data, Y_data, X_test, Y_test, black_box)
      else: 
        # Apply outlier detection method
        pvals = oc_methods[oc_name](X_data, Y_data, X_test, Y_test, black_box)       
        # Apply BH
        reject, _, _, _ = multipletests(pvals, alpha=alpha/purity_test, method='fdr_bh')

      # Evaluate FDP and Power
      rejections = np.sum(reject)
      if rejections > 0:
        fdp = np.sum(reject[np.where(Y_test==0)[0]])/reject.shape[0] 
        power = np.sum(reject[np.where(Y_test==1)[0]])/np.sum(Y_test)
      else:
        fdp = 0
        power = 0

      res_tmp = {'Method':oc_name, 'Black Box': box_name, 'Experiment': experiment, 'Dimension': p, 
                 'Purity': purity, 'Alpha':alpha, 'Rejections':rejections, 'FDR':fdp, 'Power':power}
      res_tmp = pd.DataFrame(res_tmp, index=[0])
      results = pd.concat([results, res_tmp])
  return results


## run all experiments 

random_state = 3000

p = 100     # Number of features
a =  2     # Signal amplitude (a=1 is null)
amplifier = 1.8
sep = 2


alpha = 0.1
n_experiments = 1

# Run experiments
results = pd.DataFrame()

#purity_level = np.array([0.1 ,0.2, 0.5, 0.8, 0.9])
purity_level = np.array([0.1])
for experiment in tqdm(range(n_experiments)):
  for purity in purity_level:
    # Random state for this experiment
    random_state = 2022 + int(experiment*100) + int(purity*100)
    #model = Model_Gaussian(p, sep, random_state=random_state)
    model = Model_CC(p,a,random_state=random_state)
    #print(random_state)
    res = run_experiment(model, purity, bc_methods, oc_methods, bc_bb, oc_bb, alpha=alpha, experiment=experiment, random_state=random_state)
    results = results.append(res)

# Save results on file
if out_file is not None:
    print("Saving file with {:d} rows".format(results.shape[0]))
    results.to_csv(out_file)


print("Output file for this experiment:\n  " + out_file)  
