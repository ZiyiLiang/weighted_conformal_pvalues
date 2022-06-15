import numpy as np
import statistics as stats
from arc.classification import ProbabilityAccumulator as ProbAccum


def bc(X_cal, Y_cal, X_test, Y_test, black_box):
    classifier = black_box

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



def oc(X_cal, Y_cal, X_test, Y_test, black_box):
    classifier = black_box

    # Compute scores on clean calibration data
    scores_cal = classifier.score_samples(X_cal)

    # Compute scores on test data
    scores_test = classifier.score_samples(X_test)

    # Compute conformal p-values for test points
    scores_cal_mat = np.tile(scores_cal, (len(scores_test),1))
    pvals_numerator = np.sum(scores_cal_mat <= scores_test.reshape(len(scores_test),1), 1)
    pvals_marginal = (1.0+pvals_numerator)/(1.0+len(scores_cal))
    return pvals_marginal



def clra(X_cal, Y_cal, X_test, Y_test, black_box):
    classifier = black_box
    # Calculate the scores under both hypothesis
    scores_cal0 = [classifier[0].score_samples(X_cal[0]), classifier[1].score_samples(X_cal[0])]
    scores_cal1 = [classifier[0].score_samples(X_cal[1]), classifier[1].score_samples(X_cal[1])]
    scores_test = [classifier[0].score_samples(X_test), classifier[1].score_samples(X_test)]

    n_inliers = X_cal[0].shape[0]
    n_outliers = X_cal[1].shape[0]
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
        
    # Compute the likelihood ratio for the test data
    lr_test = p1_test/(5e-10+p0_test)
    lr_calib = np.empty((n_test,n_inliers))
    for i in range(n_test):
        lr_calib[i] = p1_calib[i]/(5e-10+p0_calib[i])

    # Compute the comformal p-values
    tmp = np.sum(lr_calib >= lr_test.reshape(n_test,1), 1)
    pvals_marginal = (1.0+tmp)/(1.0+n_inliers)

    return pvals_marginal