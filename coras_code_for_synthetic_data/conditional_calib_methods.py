# Conditional calibration version of methods used in synthetic data experiments

import numpy as np
import statistics as stats

from sklearn.base import clone
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests


## CLRA with conditional calibration
def cBH(X_data, Y_data, X_test, Y_test, black_box):
  
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
  
  pvals = np.empty((n_test,n_test))
  rejections = np.empty(n_test)
  reject = np.zeros(n_test)    ## the final rejection set

  for i in range(n_test):
    # expand the calibration set by including test point i
    score0_expcal = np.append(scores_cal0[0],scores_test[0][i])
    score1_expcal = np.append(scores_cal0[1],scores_test[1][i])
      
    # compute p0 for test points and null calibration points
    # for all other test points, apply CLRA, hence each test point has its own set of calibration p values
    for j in range(n_test):
      score0_full = np.append(score0_expcal, scores_test[0][j]) if j!=i else score0_expcal
      score0_cal0 = score0_expcal if j!=i else scores_cal0[0]
      score1_cal0 = score1_expcal if j!=i else scores_cal0[1]
      n_cal0 = len(score0_cal0)
      if comp0[j] == 0:  
        p0_test = np.sum(score0_full <= scores_test[0][j])/(len(score0_full))
        p0_calib = np.sum(np.tile(score0_full, (n_cal0,1)) <= score0_cal0.reshape(n_cal0,1), 1)/(len(score0_full))    
      else:
        p0_test = np.sum(score0_full >= scores_test[0][j])/(len(score0_full))
        p0_calib = np.sum(np.tile(score0_full, (n_cal0,1)) >= score0_cal0.reshape(n_cal0,1), 1)/(len(score0_full))

      # compute p1 for test points and null calibration points
      if comp1[j] == 0:
        p1_test = (1.0 + np.sum(scores_cal1[1] <= scores_test[1][j]))/(1.0 + n_outliers)
        p1_calib = (1.0 + np.sum(np.tile(scores_cal1[1], (n_cal0,1)) <= score1_cal0.reshape(n_cal0,1), 1))/(1.0 + n_outliers)
      else:
        p1_test = (1.0 + np.sum(scores_cal1[1] >= scores_test[1][j]))/(1.0 + n_outliers)
        p1_calib = (1.0 + np.sum(np.tile(scores_cal1[1], (n_cal0,1)) >= score1_cal0.reshape(n_cal0,1), 1))/(1.0 + n_outliers)

      # calculate the conformal p-value for the other test points
      lr_calib = p0_calib/(5e-10+p1_calib)
      lr_test = p0_test/(5e-10+p1_test)
      tmp = np.sum(lr_calib <= lr_test)
      pvals[i][j] = (1.0 + tmp)/(1.0 + n_cal0)
  
 
  # determine the rejection set for each hypothesis
  for i in range(n_test):
    temp_pval = np.copy(pvals[i])
    temp_pval[i] = 0
    reject, _, _, _ = multipletests(temp_pval, alpha=alpha, method='fdr_bh')   ######## change alpha if needed
    rejections[i] = np.sum(reject)

    # Determine the final rejection set
    if pvals[i][i] <= alpha*rejections[i]/n_test:
      reject[i] = 1

  # trim the final rejection set if needed
  if np.sum((rejections > np.sum(reject))*reject) > 0:
    # randomly trim the set by generating a list of uniform r.v.s
    u = np.random.rand(n_test)
    temp_pval = u*rejections/np.sum(reject)
    # perform a secondary BH procedure on the randomized p values
    sec_reject, _, _, _ =  multipletests(temp_pval, alpha=1, method='fdr_bh')
    # Determine the final rejection set by taking the union
    untrimmed_rej = np.sum(reject)
    reject = reject * sec_reject
    print('number of cBH trimmed rejections is:', untrimmed_rej-np.sum(reject),
          'number of cBH null calibration points is:', n_inliers, '\n')  

  return reject


# conditional calibration for CLRA with leave-one-out technique
# final rejection set is determined by taking median of the loo rejection sets
def cBH_med(X_data, Y_data, X_test, Y_test, black_box):
  
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
  
  # Dertermine the direction by comparing the null and alternative scores
  # Compare the median score of inliers and outliers under the classifier 0
  tmp = [stats.median(np.append(scores_cal0[0],scores_test[0][i])) for i in range(n_test)]
  comp0 = np.array(tmp) <= np.array([stats.median(scores_cal1[0])]*n_test)

  # Compare the median score of inliers and outliers under the classifier 1
  tmp = [stats.median(np.append(scores_cal0[1],scores_test[1][i])) for i in range(n_test)]
  comp1 = np.array(tmp) >= np.array([stats.median(scores_cal1[1])]*n_test)
  
  ## Select a subset of indices to perform the loo procedure
  n_loo = 20 if (1+n_inliers) > 20 else (1+n_inliers)          #### change the sample size if needed

  pvals = np.empty((n_loo,n_test))
  loo_rej = np.empty((n_test, n_loo))
  rejections = np.empty(n_test)
  reject = np.zeros(n_test)    ## the final rejection set

  for i in range(n_test):
    score0_expcal = np.append(scores_cal0[0],scores_test[0][i])
    score1_expcal = np.append(scores_cal0[1],scores_test[1][i])
    ## randomly select the loo set
    loo = np.random.choice(1+n_inliers, n_loo, replace = False)
    for j in range(n_loo):
      # leave one point out to form the new calibration set
      loo_cal0 = np.delete(score0_expcal, loo[j])
      loo_cal1 = np.delete(score1_expcal, loo[j])  
      for k in range(n_test): 
        score0_full = np.append(loo_cal0, scores_test[0][k]) if k!=i else score0_expcal
        score0_cal0 = loo_cal0 if k!=i else scores_cal0[0]
        score1_cal0 = loo_cal1 if k!=i else scores_cal0[1] 
        n_cal0 = len(score0_cal0)
        if comp0[k] == 0:  
          p0_test = np.sum(score0_full <= scores_test[0][k])/(len(score0_full))
          p0_calib = np.sum(np.tile(score0_full, (n_cal0,1)) <= score0_cal0.reshape(n_cal0,1), 1)/(len(score0_full))    
        else:
          p0_test = np.sum(score0_full >= scores_test[0][k])/(len(score0_full))
          p0_calib = np.sum(np.tile(score0_full, (n_cal0,1)) >= score0_cal0.reshape(n_cal0,1), 1)/(len(score0_full))

        # compute p1 for test points and null calibration points
        if comp1[k] == 0:
          p1_test = (1.0 + np.sum(scores_cal1[1] <= scores_test[1][k]))/(1.0 + n_outliers)
          p1_calib = (1.0 + np.sum(np.tile(scores_cal1[1], (n_cal0,1)) <= score1_cal0.reshape(n_cal0,1), 1))/(1.0 + n_outliers)
        else:
          p1_test = (1.0 + np.sum(scores_cal1[1] >= scores_test[1][k]))/(1.0 + n_outliers)
          p1_calib = (1.0 + np.sum(np.tile(scores_cal1[1], (n_cal0,1)) >= score1_cal0.reshape(n_cal0,1), 1))/(1.0 + n_outliers)

        # calculate the conformal p-value for the other test points
        lr_calib = p0_calib/(5e-10+p1_calib)
        lr_test = p0_test/(5e-10+p1_test)
        tmp = np.sum(lr_calib <= lr_test)
        pvals[j][k] = (1.0 + tmp)/(1.0 + n_cal0)

    # determine the rejection set for each hypothesis
    for j in range(n_loo):
      temp_pval = np.copy(pvals[j])
      temp_pval[i] = 0
      reject, _, _, _ = multipletests(temp_pval, alpha=alpha, method='fdr_bh')   ######## change alpha if needed
      loo_rej[i][j] = np.sum(reject)

    # Determine the final rejection set
    loo_idx = (np.abs(loo_rej[i]-np.median(loo_rej[i]))).argmin()
    rejections[i]=loo_rej[i][loo_idx]
    if pvals[loo_idx][i] <= alpha*rejections[i]/n_test:
      reject[i] = 1


  # trim the final rejection set if needed
  if np.sum((rejections > np.sum(reject))*reject) > 0:
    # randomly trim the set by generating a list of uniform r.v.s
    u = np.random.rand(n_test)
    temp_pval = u*rejections/np.sum(reject)
    # perform a secondary BH procedure on the randomized p values
    sec_reject, _, _, _ =  multipletests(temp_pval, alpha=1, method='fdr_bh')   ######## change alpha if needed
    # Determine the final rejection set by taking the union
    untrimmed_rej = np.sum(reject)
    reject = reject * sec_reject
    print('number of cBH_med trimmed rejections is:', untrimmed_rej-np.sum(reject),
          'number of cBH_med null calibration points is:', n_inliers, '\n')
  
  return reject




## CLRA_CV++ with conditional calibration
def clra_cvpp_cBH(X_data, Y_data, X_test, Y_test, black_box, nfold=3): 
  ###### change the fdr level or sparsity level if needed
  alpha = 0.1/0.5

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

  # for each test point, p-values for other test points are calculated differently
  pvals = np.empty((n_test, n_test))
  rejections = np.empty(n_test)
  reject = np.zeros(n_test)     # the final rejection set

  # split the cv sets
  for i in range(n_test):
    for j in range(n_test):
      # append the test point to the null set before splitting           #### might want to fix the splitting by a same random state for all test points
      X_full, n_full = np.append(X_inliers, [X_test[i]], axis=0), n_inliers + 1
      # for all j != i, that is, for all other test points, expand the calibration set by combinining test point i
      if j != i: X_full, n_full = np.append(X_full, [X_test[j]], axis=0), n_inliers + 2

      cv0 = KFold(n_splits=nfold, shuffle=True)
      cv1 = KFold(n_splits=nfold, shuffle=True)
      cv0_list = list(cv0.split(np.arange(n_full)))
      cv1_list = list(cv1.split(np.arange(n_outliers)))
      ## record the index ordering in the folds to retrieve the test point
      infold_idx = np.concatenate([split[1] for split in cv0_list])
      X_ordered = X_full[infold_idx]

      score0_full, score1_full, score1_cal1 = [],[[]]*nfold,[[]]*nfold
      for fold in range(nfold):
        # list of training set
        X_train = [X_full[cv0_list[fold][0]], X_outliers[cv1_list[fold][0]]]
        X_calib = [X_full[cv0_list[fold][1]], X_outliers[cv1_list[fold][1]]]
        
        # train the classifiers for both inliers and outliers
        classifier = [[]]*2
        for k in range(2):
          classifier[k] = clone(black_box)
          classifier[k].fit(X_train[k])

        ## calculate the scores for class-0
        tmp_score0_full = classifier[0].score_samples(X_calib[0])
        score0_full = tmp_score0_full if not len(score0_full) else np.append(score0_full, tmp_score0_full)

        ## calculate the scores for class-1
        score1_full[fold] = classifier[1].score_samples(X_ordered)
        score1_cal1[fold] = classifier[1].score_samples(X_calib[1])

       # calculate p0 for the augmented data set
      if comp0[j] == 0:
        p0_full = np.sum(np.tile(score0_full, (n_full, 1)) <= score0_full.reshape(n_full, 1), 1)/n_full
      else:
        p0_full = np.sum(np.tile(score0_full, (n_full, 1)) >= score0_full.reshape(n_full, 1), 1)/n_full
      # calculate p1 by traversing through all the folds
      tmp = 0
      for fold in range(nfold):
        if comp1[j] == 0:
          tmp = tmp + np.sum(np.tile(score1_cal1[fold], (n_full, 1)) <= score1_full[fold].reshape(n_full,1), 1)
        else:
          tmp = tmp + np.sum(np.tile(score1_cal1[fold], (n_full, 1)) >= score1_full[fold].reshape(n_full,1), 1)
      p1_full = (1.0 + tmp)/(1.0 + n_outliers)
      # calculate the LR for the augmented data set
      lr_full = p0_full/(5e-10 + p1_full)
      
      ## retrieve the test point from the augmented set
      ## and calculate the marginal p-value
      idx_test = np.where(infold_idx == n_full-1)[0]
      lr_test = lr_full[idx_test]
      pvals[i][j] = np.sum(lr_full <= lr_test)/(1 + n_inliers)

  # determine the rejection set for each hypothesis
  for i in range(n_test):
    temp_pval = np.copy(pvals[i])
    temp_pval[i] = 0
    reject, _, _, _ = multipletests(temp_pval, alpha=alpha, method='fdr_bh')   ######## change alpha if needed
    rejections[i] = np.sum(reject)

    # Determine the final rejection set
    if pvals[i][i] <= alpha*rejections[i]/n_test:
      reject[i] = 1

  # trim the final rejection set if needed
  if np.sum((rejections > np.sum(reject))*reject) > 0:
    # randomly trim the set by generating a list of uniform r.v.s
    u = np.random.rand(n_test)
    temp_pval = u*rejections/np.sum(reject)
    # perform a secondary BH procedure on the randomized p values
    sec_reject, _, _, _ =  multipletests(temp_pval, alpha=1, method='fdr_bh')   ######## change alpha if needed
    # Determine the final rejection set by taking the union
    untrimmed_rej = np.sum(reject)
    reject = reject * sec_reject
    print('number of cBH trimmed rejections is:', untrimmed_rej-np.sum(reject),
          'number of cBH null calibration points is:', n_inliers, '\n')
  return reject



## CLRA_CV++ with leave-one-out conditional calibration
def clra_cvpp_cBH_loo(X_data, Y_data, X_test, Y_test, black_box, nfold=3): 
  ###### change the fdr level or sparsity level if needed
  alpha = 0.1/0.5

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


  ## Select a subset of indices to perform the loo procedure
  n_loo = 10 if (1+n_inliers) > 10 else (1+n_inliers)          #### change the sample size if needed

  # for each test point, p-values for other test points are calculated differently
  pvals = np.empty((n_loo, n_test))
  loo_rej = np.empty((n_test, n_loo))
  rejections = np.empty(n_test)
  reject = np.zeros(n_test)     # the final rejection set

  # split the cv sets
  for i in range(n_test):
    loo_idx = np.random.choice(1+n_inliers, n_loo, replace = False)
    for loo in range(n_loo):
      # append the test point to the null set before splitting         
      X_full, n_full = np.append(X_inliers, [X_test[i]], axis=0), n_inliers + 1
      for j in range(n_test):       
        ## for all j != i, that is, for all other test points, formulated the loo set
        ## by randomly discrading one observation from the expanded calibration set
        if j != i: 
          X_loo = np.delete(X_full, loo_idx[loo], axis=0)
          X_loo, n_full = np.append(X_loo, [X_test[j]], axis=0), n_inliers + 1
        else:
          X_loo = X_full

        cv0 = KFold(n_splits=nfold, shuffle=True)
        cv1 = KFold(n_splits=nfold, shuffle=True)
        cv0_list = list(cv0.split(np.arange(n_full)))
        cv1_list = list(cv1.split(np.arange(n_outliers)))
        ## record the index ordering in the folds to retrieve the test point
        infold_idx = np.concatenate([split[1] for split in cv0_list])
        X_ordered = X_loo[infold_idx]

        score0_full, score1_full, score1_cal1 = [],[[]]*nfold,[[]]*nfold
        for fold in range(nfold):
          # list of training set
          X_train = [X_loo[cv0_list[fold][0]], X_outliers[cv1_list[fold][0]]]
          X_calib = [X_loo[cv0_list[fold][1]], X_outliers[cv1_list[fold][1]]]
          
          # train the classifiers for both inliers and outliers
          classifier = [[]]*2
          for k in range(2):
            classifier[k] = clone(black_box)
            classifier[k].fit(X_train[k])

          ## calculate the scores for class-1
          score1_full[fold] = classifier[1].score_samples(X_ordered)
          score1_cal1[fold] = classifier[1].score_samples(X_calib[1])

          ## calculate the scores for class-1
          tmp_score1_full = classifier[1].score_samples(X_calib[0])
          tmp_score1_cal1 = classifier[1].score_samples(X_calib[1])
          score1_full = tmp_score1_full if not len(score1_full) else np.append(score1_full, tmp_score1_full)
          score1_cal1 = tmp_score1_cal1 if not len(score1_cal1) else np.append(score1_cal1, tmp_score1_cal1)

        # calculate p0 for the augmented data set
        if comp0[j] == 0:
          p0_full = np.sum(np.tile(score0_full, (n_full, 1)) <= score0_full.reshape(n_full, 1), 1)/n_full
        else:
          p0_full = np.sum(np.tile(score0_full, (n_full, 1)) >= score0_full.reshape(n_full, 1), 1)/n_full
        # calculate p1 by traversing through all the folds
        tmp = 0
        for fold in range(nfold):
          if comp1[j] == 0:
            tmp = tmp + np.sum(np.tile(score1_cal1[fold], (n_full, 1)) <= score1_full[fold].reshape(n_full,1), 1)
          else:
            tmp = tmp + np.sum(np.tile(score1_cal1[fold], (n_full, 1)) >= score1_full[fold].reshape(n_full,1), 1)
        p1_full = (1.0 + tmp)/(1.0 + n_outliers)
        # calculate the LR for the augmented data set
        lr_full = p0_full/(5e-10 + p1_full)
        
        ## retrieve the test point from the augmented set
        ## and calculate the marginal p-value
        idx_test = np.where(infold_idx == n_full - 1)[0]
        lr_test = lr_full[idx_test]
        pvals[loo][j] = np.sum(lr_full <= lr_test)/(1 + n_inliers)
    
    # determine the rejection set for each loo set
    for loo in range(n_loo):
      temp_pval = np.copy(pvals[loo])
      temp_pval[i] = 0
      reject, _, _, _ = multipletests(temp_pval, alpha=alpha, method='fdr_bh')   ######## change alpha if needed
      loo_rej[i][loo] = np.sum(reject)

    # Determine the final rejection set
    med_idx = (np.abs(loo_rej[i]-np.median(loo_rej[i]))).argmin()
    rejections[i]=loo_rej[i][med_idx]
    if pvals[med_idx][i] <= alpha*rejections[i]/n_test:
      reject[i] = 1

  # trim the final rejection set if needed
  if np.sum((rejections > np.sum(reject))*reject) > 0:
    # randomly trim the set by generating a list of uniform r.v.s
    u = np.random.rand(n_test)
    temp_pval = u*rejections/np.sum(reject)
    # perform a secondary BH procedure on the randomized p values
    sec_reject, _, _, _ =  multipletests(temp_pval, alpha=1, method='fdr_bh')   ######## change alpha if needed
    # Determine the final rejection set by taking the union
    untrimmed_rej = np.sum(reject)
    reject = reject * sec_reject
    print('number of cBH_cv_med trimmed rejections is:', untrimmed_rej-np.sum(reject),
          'number of cBH_cv_med null calibration points is:', n_inliers, '\n')
    #print('number of null calibration points is:', n_inliers, '\n')

  return reject