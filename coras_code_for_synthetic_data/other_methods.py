import numpy as np
import statistics as stats
from sklearn.model_selection import KFold

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests


# OC with cross validation, nfold determines the number of splits
def oc_cvplus(X_data, Y_data, X_test, Y_test, black_box, nfold=3): 
  n_test = X_test.shape[0]
  idx_inliers = np.where(Y_data==0)[0]

  scores_test = np.empty((nfold,n_test))
  scores_cal0 = [[]]*nfold
  # split the cv sets
  cv0 = KFold(n_splits=nfold, shuffle=True)
  cv0_list = list(cv0.split(idx_inliers))
  for fold in range(nfold):
    # list of training set for inliers
    X_train = X_data[idx_inliers[cv0_list[fold][0]]]
    X_calib = X_data[idx_inliers[cv0_list[fold][1]]]
    
    # train the classifier
    classifier = clone(black_box)
    classifier.fit(X_train)

    # calculate the scores
    scores_cal0[fold] = classifier.score_samples(X_calib)
    scores_test[fold] = classifier.score_samples(X_test)

  # compute the marginal p values by combining all folds
  tmp = 0
  for i in range(nfold):
    tmp = tmp + np.sum(np.tile(scores_cal0[i], (n_test, 1)) <= scores_test[i].reshape(n_test,1), 1)
  pvals = (1.0 + tmp)/(1.0 + idx_inliers.shape[0])

  return pvals


# OC with CV++, mix the test point to null set before sample splitting
def oc_cvpp(X_data, Y_data, X_test, Y_test, black_box, nfold=3): 
  n_test = X_test.shape[0]
  idx_inliers = np.where(Y_data==0)[0]
  X_inliers = X_data[idx_inliers]
  n_inliers = idx_inliers.shape[0]

  scores_test = np.empty(n_test)
  scores_cal0 = [[]]*n_test
  # split the cv sets
  for i in range(n_test):
    # append the test point to the null set before splitting
    X_full = np.append(X_inliers, [X_test[i]], axis=0)
    cv0 = KFold(n_splits=nfold, shuffle=True)
    cv0_list = list(cv0.split(np.arange(n_inliers + 1)))
    for fold in range(nfold):
      # list of training set
      X_train = X_full[cv0_list[fold][0]]
      X_calib = X_full[cv0_list[fold][1]]
      
      # train the classifier
      classifier = clone(black_box)
      classifier.fit(X_train)

      # calculate the scores
      tmp_cal_scores = classifier.score_samples(X_calib)
      if not len(scores_cal0[i]):
        scores_cal0[i] = tmp_cal_scores
      else:
        scores_cal0[i] = np.append(scores_cal0[i], tmp_cal_scores)
      
      # if test point is in this calibration fold, update the test score    
      if n_inliers in cv0_list[fold][1]:
        scores_test[i] = classifier.score_samples(np.array([X_test[i]]))

  # compute the marginal p values by combining all folds
  pvals_numerator = np.sum(scores_cal0 <= scores_test.reshape(n_test, 1), 1)
  pvals =  pvals_numerator/(1.0 + n_inliers)

  return pvals


## clra without mixing the additional test point
## test function to see the effect on power by enlarging the calibration set
def clra_oracle(X_data, Y_data, X_test, Y_test, black_box):
  
  X_train = [[]]*2
  X_calib = [[]]*2
  Y_train = [[]]*2
  Y_calib = [[]]*2
  classifier = [[]]*2

  # fit one-class black-box method and calibrate for each label
  for i in range(2):
    X_train[i], X_calib[i], Y_train[i], Y_calib[i] = train_test_split(X_data[np.where(Y_data==i)[0]], Y_data[np.where(Y_data==i)[0]], test_size=0.5)

    # Train a black-box one-class classifier
    #classifier[i] = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.1)
    classifier[i] = clone(black_box)

    # Fit the black-box one-class classifier
    classifier[i].fit(X_train[i])

  # Calculate the scores under both hypothesis
  scores_cal0 = [classifier[0].score_samples(X_calib[0]), classifier[1].score_samples(X_calib[0])]
  scores_cal1 = [classifier[0].score_samples(X_calib[1]), classifier[1].score_samples(X_calib[1])]
  scores_test = [classifier[0].score_samples(X_test), classifier[1].score_samples(X_test)]

  n_inliers = X_calib[0].shape[0]
  n_outliers = X_calib[1].shape[0]
  n_test = X_test.shape[0]

  # Compare the median score of inliers and outliers under the classifier 0
  tmp = [stats.median(np.append(scores_cal0[0],scores_test[0][i])) for i in range(n_test)]
  comp0 = np.array(tmp) <= np.array([stats.median(scores_cal1[0])]*n_test)

  # Compare the median score of inliers and outliers under the classifier 1
  tmp = [stats.median(np.append(scores_cal0[1],scores_test[1][i])) for i in range(n_test)]
  comp1 = np.array(tmp) >= np.array([stats.median(scores_cal1[1])]*n_test)
 
  # Compute the ratio for the test data and null calibration data
  # automatically choose the quantile by the comparison result above
  p0_test = np.empty(n_test)
  p1_test = np.empty(n_test)
  p0_calib = np.empty((n_test,n_inliers))
  p1_calib = np.empty((n_test,n_inliers))

  # Compute p0 for test points and null calibration points
  for i in range(n_test):
    if comp0[i] == 0:
      p0_test[i] = np.sum(scores_cal0[0] <= scores_test[0][i])/n_inliers
      p0_calib[i] = np.sum(np.tile(scores_cal0[0], (n_inliers,1)) <= scores_cal0[0].reshape(n_inliers,1), 1)/n_inliers
    else:
      p0_test[i] = np.sum(scores_cal0[0] >= scores_test[0][i])/n_inliers
      p0_calib[i] = np.sum(np.tile(scores_cal0[0], (n_inliers,1)) >= scores_cal0[0].reshape(n_inliers,1), 1)/n_inliers

  # Compute p1 for test points and null calibration points
  for i in range(n_test):
    if comp1[i] == 0:
      p1_test[i] = np.sum(scores_cal1[1] <= scores_test[1][i])/n_outliers
      p1_calib[i] = np.sum(np.tile(scores_cal1[1], (n_inliers,1)) <= scores_cal0[1].reshape(n_inliers,1), 1)/n_outliers
    else:
      p1_test[i] = np.sum(scores_cal1[1] >= scores_test[1][i])/n_outliers
      p1_calib[i] = np.sum(np.tile(scores_cal1[1], (n_inliers,1)) >= scores_cal0[1].reshape(n_inliers,1), 1)/n_outliers
  
  # Compute the ratio for the test data
  lr_test = p0_test/(5e-10+p1_test)
  lr_calib = np.empty((n_test,n_inliers))
  for i in range(n_test):
    lr_calib[i] = p0_calib[i]/(5e-10+p1_calib[i])

  # Compute the comformal p-values
  tmp = np.sum(lr_calib <= lr_test.reshape(n_test,1), 1)
  pvals = (1.0+tmp)/(1.0+n_inliers)

  return pvals


# oracle conditional calibration methods for CLRA
# oracle procedure that never triggers the trimming process
# however this oracle function will not produce rejection sets with valid fdr control,
# it is used to see the power impact of mixing the extra test point into the calibration set
def cBH_oracle(X_data, Y_data, X_test, Y_test, black_box):
  
  ##### change the fdr level if needed
  alpha = 0.1/0.5
  X_train = [[]]*2
  X_calib = [[]]*2
  Y_train = [[]]*2
  Y_calib = [[]]*2
  classifier = [[]]*2

  # fit one-class black-box method and calibrate for each label
  for i in range(2):
    X_train[i], X_calib[i], Y_train[i], Y_calib[i] = train_test_split(X_data[np.where(Y_data==i)[0]], Y_data[np.where(Y_data==i)[0]], test_size=0.5)

    # Train a black-box one-class classifier
    #classifier[i] = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.1)
    classifier[i] = clone(black_box)

    # Fit the black-box one-class classifier
    classifier[i].fit(X_train[i])

  # Calculate the scores under both hypothesis
  scores_cal0 = [classifier[0].score_samples(X_calib[0]), classifier[1].score_samples(X_calib[0])]
  scores_cal1 = [classifier[0].score_samples(X_calib[1]), classifier[1].score_samples(X_calib[1])]
  scores_test = [classifier[0].score_samples(X_test), classifier[1].score_samples(X_test)]

  n_test = X_test.shape[0]
  n_inliers = X_calib[0].shape[0]
  n_outliers = X_calib[1].shape[0]
  
  ## Dertermine the direction by comparing the null and alternative scores
  # Compare the median score of inliers and outliers under the classifier 0
  tmp = [stats.median(np.append(scores_cal0[0],scores_test[0][i])) for i in range(n_test)]
  comp0 = np.array(tmp) <= np.array([stats.median(scores_cal1[0])]*n_test)

  # Compare the median score of inliers and outliers under the classifier 1
  tmp = [stats.median(np.append(scores_cal0[1],scores_test[1][i])) for i in range(n_test)]
  comp1 = np.array(tmp) >= np.array([stats.median(scores_cal1[1])]*n_test)

  ## Compute the rejection set for each test point
  p0_test = np.empty(n_test)
  p1_test = np.empty(n_test)
  p0_calib = np.empty((n_test,n_inliers))
  p1_calib = np.empty((n_test,n_inliers))

  # Compute p0 for test points and null calibration points
  for i in range(n_test):
    if comp0[i] == 0:
      p0_test[i] = np.sum(scores_cal0[0] <= scores_test[0][i])/n_inliers
      p0_calib[i] = np.sum(np.tile(scores_cal0[0], (n_inliers,1)) <= scores_cal0[0].reshape(n_inliers,1), 1)/n_inliers
    else:
      p0_test[i] = np.sum(scores_cal0[0] >= scores_test[0][i])/n_inliers
      p0_calib[i] = np.sum(np.tile(scores_cal0[0], (n_inliers,1)) >= scores_cal0[0].reshape(n_inliers,1), 1)/n_inliers

  # Compute p1 for test points and null calibration points
  for i in range(n_test):
    if comp1[i] == 0:
      p1_test[i] = np.sum(scores_cal1[1] <= scores_test[1][i])/n_outliers
      p1_calib[i] = np.sum(np.tile(scores_cal1[1], (n_inliers,1)) <= scores_cal0[1].reshape(n_inliers,1), 1)/n_outliers
    else:
      p1_test[i] = np.sum(scores_cal1[1] >= scores_test[1][i])/n_outliers
      p1_calib[i] = np.sum(np.tile(scores_cal1[1], (n_inliers,1)) >= scores_cal0[1].reshape(n_inliers,1), 1)/n_outliers
  
  # Compute the ratio for the test data
  lr_test = p0_test/(5e-10+p1_test)
  lr_calib = np.empty((n_test,n_inliers))
  for i in range(n_test):
    lr_calib[i] = p0_calib[i]/(5e-10+p1_calib[i])

  # Compute the comformal p-values
  tmp = np.sum(lr_calib < lr_test.reshape(n_test,1), 1) +\
        np.ceil(np.sum(lr_calib == lr_test.reshape(n_test,1), 1) * np.random.rand(n_test))
  pvals = tmp/n_inliers
  
  # determine the rejection set for each hypothesis
  rejections = np.empty(n_test)
  for i in range(n_test):
    temp_pval = np.copy(pvals)
    temp_pval[i] = 0
    reject, _, _, _ = multipletests(temp_pval, alpha=alpha, method='fdr_bh')   ######## change alpha if needed
    rejections[i] = np.sum(reject)

  # Determine the final rejection set
  reject = np.zeros(n_test)
  for i in range(n_test):
    if pvals[i] <= alpha*rejections[i]/n_test:
      reject[i] = 1

  clra_rej, _, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
  print('normal BH rejections: ', np.sum(clra_rej), 'cBH rejections: ', np.sum(reject))
  return reject

