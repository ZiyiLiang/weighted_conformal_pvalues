import numpy as np

def conformalize_scores(scores_cal, scores_test, offset=1):
    '''Compute the conformal p-values.

    Parameters:
    -----------
    scores_cal: array_like
                The calibration scores.
    scores_test:array_like
                The test scores.
    offset:     int
                Value needs to be 0 or 1.
    
    Returns:
    --------
    pvals:      ndarray
                The conformal p-values.
                     
    '''
    assert((offset==0) or (offset==1))
    n_cal = len(scores_cal) - (1-offset)
    scores_mat = np.tile(scores_cal, (len(scores_test),1))
    tmp = np.sum(scores_mat <= scores_test.reshape(len(scores_test),1), 1)
    pvals = (offset+tmp)/(1.0+n_cal)
    return pvals
