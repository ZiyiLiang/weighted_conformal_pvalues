# Main methods used in synthetic data experiments
import sys
sys.path.append("/mnt/c/users/liang/OneDrive/Desktop/CLRA/arc/")
sys.path.append('/mnt/c/users/liang/OneDrive/Desktop/CLRA/cqr-comparison')
sys.path.append('/mnt/c/users/liang/OneDrive/Desktop/CLRA/cqr')
sys.path.append('/mnt/c/users/liang/OneDrive/Desktop/CLRA/conditional-conformal-pvalues')
import numpy as np
import statistics as stats
from sklearn.model_selection import KFold

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from arc.classification import ProbabilityAccumulator as ProbAccum

def bc(X_data, Y_data, X_test, Y_test, black_box):
  # Divide the data into training and calibration sets,
  # making sure the calibration set has no outliers
  idx_inliers = np.where(Y_data==0)[0]
  idx_outliers = np.where(Y_data==1)[0]
  idx_train, idx_cal = train_test_split(idx_inliers, test_size=0.5)
  idx_train = np.concatenate([idx_train, idx_outliers])
  np.random.shuffle(idx_train)
  X_train = X_data[idx_train]
  Y_train = Y_data[idx_train]
  X_cal = X_data[idx_cal]
  Y_cal = Y_data[idx_cal]

  # Fit model
  classifier = clone(black_box)
  classifier.fit(X_train, Y_train)

  # Compute conformity scores on calibration data
  n_cal = X_cal.shape[0]
  pi_hat_cal = classifier.predict_proba(X_cal)
  grey_box = ProbAccum(pi_hat_cal)
  epsilon_cal = np.random.uniform(low=0.0, high=1.0, size=n_cal)
  scores_cal = grey_box.calibrate_scores(Y_cal, epsilon=epsilon_cal)

  # Compute conformity scores on test data, under the null Y = 0
  Y_test_placeholder = np.zeros(Y_test.shape).astype(int)
  n_test = X_test.shape[0]
  pi_hat_test = classifier.predict_proba(X_test)
  grey_box_test = ProbAccum(pi_hat_test)
  epsilon_test = np.random.uniform(low=0.0, high=1.0, size=n_test)
  scores_test = grey_box_test.calibrate_scores(Y_test_placeholder, epsilon=epsilon_test)

  # Compute conformal p-values for test points
  scores_cal_mat = np.tile(scores_cal, (len(scores_test),1))
  pvals_numerator = np.sum(scores_cal_mat <= scores_test.reshape(len(scores_test),1), 1)
  pvals_marginal = (1.0+pvals_numerator)/(1.0+len(scores_cal))
  
  return pvals_marginal


# Divide the data into training and calibration sets,
# making sure neither the training nor the calibration sets have any outliers
def oc(X_data, Y_data, X_test, Y_test, black_box):
  idx_inliers = np.where(Y_data==0)[0]
  idx_outliers = np.where(Y_data==1)[0]
  idx_train, idx_cal = train_test_split(idx_inliers, test_size=0.5)
  X_train = X_data[idx_train]
  Y_train = Y_data[idx_train]
  X_cal = X_data[idx_cal]
  Y_cal = Y_data[idx_cal]

  # Fit the black-box one-class classifier
  classifier = clone(black_box)
  classifier.fit(X_train)

  # Compute scores on clean calibration data
  scores_cal = classifier.score_samples(X_cal)

  # Compute scores on test data
  scores_test = classifier.score_samples(X_test)

  # Compute conformal p-values for test points
  scores_cal_mat = np.tile(scores_cal, (len(scores_test),1))
  pvals_numerator = np.sum(scores_cal_mat <= scores_test.reshape(len(scores_test),1), 1)
  pvals_marginal = (1.0+pvals_numerator)/(1.0+len(scores_cal))

  return pvals_marginal


# Algorithm 1, weighted p-values without auto area selection
def clr(X_data, Y_data, X_test, Y_test, black_box):
  
  X_train = [[]]*2
  X_calib = [[]]*2
  Y_train = [[]]*2
  Y_calib = [[]]*2
  classifier = [[]]*2

  # fit one-class black-box method and calibrate for each label
  for i in range(2):
    X_train[i], X_calib[i], Y_train[i], Y_calib[i] = train_test_split(X_data[np.where(Y_data==i)[0]], Y_data[np.where(Y_data==i)[0]], test_size=0.2)

    # Train a black-box one-class classifier
    classifier[i] = clone(black_box)

    # Fit the black-box one-class classifier
    classifier[i].fit(X_train[i])

  # Calculate the scores under both hypothesis
  scores_cal0 = [classifier[0].score_samples(X_calib[0]), classifier[1].score_samples(X_calib[0])]
  scores_cal1 = [classifier[0].score_samples(X_calib[1]), classifier[1].score_samples(X_calib[1])]
  scores_test = [classifier[0].score_samples(X_test), classifier[1].score_samples(X_test)]

  # Compute the p values for the inlier calibration data under both hypotheses
  n_inliers = X_calib[0].shape[0]
  n_outliers = X_calib[1].shape[0]
  n_test = X_test.shape[0]
  
  p0_test = np.empty(n_test)
  p1_test = np.empty(n_test)
  p0_calib = np.empty((n_test,n_inliers))
  p1_calib = np.empty((n_test,n_inliers))

  for i in range(n_test):
    # Include the test point to null calibration set for exchangeability
    # Expanded null calibration set
    scores_expcal = np.append(scores_cal0[0],scores_test[0][i])
    p0_test[i] = np.sum(scores_expcal <= scores_test[0][i])/(1.0 + n_inliers)
    p0_calib[i] = np.sum(np.tile(scores_expcal, (n_inliers,1)) <= scores_cal0[0].reshape(n_inliers,1), 1)/(1.0 + n_inliers)
  
    p1_test[i] = (1.0 + np.sum(scores_cal1[1] <= scores_test[1][i]))/(1.0 + n_outliers)
    p1_calib[i] = (1.0 + np.sum(np.tile(scores_cal1[1], (n_inliers,1)) <= scores_cal0[1].reshape(n_inliers,1), 1))/(1.0 + n_outliers)
  
  # Compute the ratio for the test data
  lr_test = p0_test/(5e-10+p1_test)
  lr_calib = np.empty((n_test,n_inliers))
  for i in range(n_test):
    lr_calib[i] = p0_calib[i]/(5e-10+p1_calib[i])
  
  # Compute the comformal p-values
  tmp = np.sum(lr_calib <= lr_test.reshape(n_test,1), 1)
  pvals_marginal = (1.0+tmp)/(1.0+n_inliers)

  return pvals_marginal




## CLR with auto area selection
def clra(X_data, Y_data, X_test, Y_test, black_box):
  
  X_train = [[]]*2
  X_calib = [[]]*2
  Y_train = [[]]*2
  Y_calib = [[]]*2
  classifier = [[]]*2

  # fit one-class black-box method and calibrate for each label
  for i in range(2):
    X_train[i], X_calib[i], Y_train[i], Y_calib[i] = train_test_split(X_data[np.where(Y_data==i)[0]], Y_data[np.where(Y_data==i)[0]], test_size=0.2)

    # Train a black-box one-class classifier
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

  # Compute the likelihood ratio for the test data and null calibration data
  # automatically choose the quantile by the comparison result above
  p0_test = np.empty(n_test)
  p1_test = np.empty(n_test)
  p0_calib = np.empty((n_test,n_inliers))
  p1_calib = np.empty((n_test,n_inliers))

  # Compute p0 for test points and null calibration points
  for i in range(n_test):
    # Include the test point to null calibration set for exchangeability
    # Expanded null calibration set
    scores_expcal = np.append(scores_cal0[0],scores_test[0][i])
    if comp0[i] == 0:
      p0_test[i] = np.sum(scores_expcal <= scores_test[0][i])/(1.0 + n_inliers)
      p0_calib[i] = np.sum(np.tile(scores_expcal, (n_inliers,1)) <= scores_cal0[0].reshape(n_inliers,1), 1)/(1.0 + n_inliers)
    else:
      p0_test[i] = np.sum(scores_expcal >= scores_test[0][i])/(1.0 + n_inliers)
      p0_calib[i] = np.sum(np.tile(scores_expcal, (n_inliers,1)) >= scores_cal0[0].reshape(n_inliers,1), 1)/(1.0 + n_inliers)

  # Compute p1 for test points and null calibration points
  for i in range(n_test):
    if comp1[i] == 0:
      p1_test[i] = (1.0 + np.sum(scores_cal1[1] <= scores_test[1][i]))/(1.0 + n_outliers)
      p1_calib[i] = (1.0 + np.sum(np.tile(scores_cal1[1], (n_inliers,1)) <= scores_cal0[1].reshape(n_inliers,1), 1))/(1.0 + n_outliers)
    else:
      p1_test[i] = (1.0 + np.sum(scores_cal1[1] >= scores_test[1][i]))/(1.0 + n_outliers)
      p1_calib[i] = (1.0 + np.sum(np.tile(scores_cal1[1], (n_inliers,1)) >= scores_cal0[1].reshape(n_inliers,1), 1))/(1.0 + n_outliers)
  
  # Compute the ratio for the test data
  lr_test = p0_test/(5e-10+p1_test)
  lr_calib = np.empty((n_test,n_inliers))
  for i in range(n_test):
    lr_calib[i] = p0_calib[i]/(5e-10+p1_calib[i])

  # Compute the comformal p-values
  tmp = np.sum(lr_calib <= lr_test.reshape(n_test,1), 1)
  pvals_marginal = (1.0+tmp)/(1.0+n_inliers)

  return pvals_marginal


## CLRA with CV++
def clra_cvpp(X_data, Y_data, X_test, Y_test, black_box, nfold=3): 
  n_test = X_test.shape[0]
  idx_inliers = np.where(Y_data==0)[0]
  idx_outliers = np.where(Y_data==1)[0]
  X_inliers = X_data[idx_inliers]
  X_outliers = X_data[idx_outliers]
  n_inliers = idx_inliers.shape[0]
  n_outliers = idx_outliers.shape[0]

  # used for auto area selection
  comp0 = np.empty(n_test)
  comp1 = np.empty(n_test)
  # train the black-box on outliers
  classifier1 = black_box
  classifier1.fit(X_data[idx_outliers])
  for i in range(n_test):
    # train the black-box on inliers
    classifier0 = black_box
    # to meet exchangbility, append test point to the null sets
    X_full = np.append(X_data[idx_inliers], [X_test[i]], axis=0)
    classifier0.fit(X_full)
    comp0[i] = stats.median(classifier0.score_samples(X_full)) <= stats.median(classifier0.score_samples(X_data[idx_outliers]))
    comp1[i] = stats.median(classifier1.score_samples(X_full)) >= stats.median(classifier1.score_samples(X_data[idx_outliers]))


  # store the p-values for all test points
  pvals = np.empty(n_test)
  # split the cv sets
  for i in range(n_test):
    # append the test point to the null set before splitting
    X_full, n_full = np.append(X_inliers, [X_test[i]], axis=0), 1 + n_inliers
    cv0 = KFold(n_splits=nfold, shuffle=True)
    cv1 = KFold(n_splits=nfold, shuffle=True)
    cv0_list = list(cv0.split(np.arange(n_full)))
    cv1_list = list(cv1.split(np.arange(n_outliers)))
    ## record the indices used for calibration in the folds to retrieve the ordering of the augmented set
    infold_idx = np.concatenate([split[1] for split in cv0_list])
    X_ordered = X_full[infold_idx]

    score0_full, score1_full, score1_cal1 = [],[[]]*nfold,[[]]*nfold
    for fold in range(nfold):
      # list of training set
      X_train = [X_full[cv0_list[fold][0]], X_outliers[cv1_list[fold][0]]]
      X_calib = [X_full[cv0_list[fold][1]], X_outliers[cv1_list[fold][1]]]
      
      # train the classifiers for both inliers and outliers
      classifier = [[]]*2
      for j in range(2):
        classifier[j] = clone(black_box)
        classifier[j].fit(X_train[j])

      # calculate the scores for class-0 
      tmp_score0_full = classifier[0].score_samples(X_calib[0])
      score0_full = tmp_score0_full if not len(score0_full) else np.append(score0_full, tmp_score0_full)

      # calculate the scores for class-1
      score1_full[fold] = classifier[1].score_samples(X_ordered)
      score1_cal1[fold] = classifier[1].score_samples(X_calib[1])
  
    # calculate p0 for the augmented data set
    if comp0[i] == 0:
      p0_full = np.sum(np.tile(score0_full, (n_full, 1)) <= score0_full.reshape(n_full, 1), 1)/n_full
    else:
      p0_full = np.sum(np.tile(score0_full, (n_full, 1)) >= score0_full.reshape(n_full, 1), 1)/n_full
    # calculate p1 by traversing through all the folds
    tmp = 0
    for fold in range(nfold):
      if comp1[i] == 0:
        tmp = tmp + np.sum(np.tile(score1_cal1[fold], (n_full, 1)) <= score1_full[fold].reshape(n_full,1), 1)
      else:
        tmp = tmp + np.sum(np.tile(score1_cal1[fold], (n_full, 1)) >= score1_full[fold].reshape(n_full,1), 1)
    p1_full = (1.0 + tmp)/(1.0 + n_outliers)
    # calculate the LR for the augmented data set
    lr_full = p0_full/(5e-10 + p1_full)
    
    # retrieve the test point from the augmented set
    # and calculate the marginal p-value
    idx_test = np.where(infold_idx == n_full-1)[0]
    lr_test = lr_full[idx_test]
    pvals[i] = np.sum(lr_full <= lr_test)/(1 + n_inliers)

  return pvals