import numpy as np
import copy
import sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import warnings
import pdb

sys.path.append('../third_party')

# Compute conformal p-values
def conformalize_scores(scores_cal, scores_test, offset=1):
    assert((offset==0) or (offset==1))
    # Calculate conformal p-values
    n_cal = len(scores_cal) - (1-offset)
    scores_mat = np.tile(scores_cal, (len(scores_test),1))
    tmp = np.sum(scores_mat <= scores_test.reshape(len(scores_test),1), 1)
    pvals = (offset+tmp)/(1.0+n_cal)
    return pvals
