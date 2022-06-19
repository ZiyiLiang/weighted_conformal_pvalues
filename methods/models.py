import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from scipy import optimize
import scipy.special as sp

import pdb

class DataModel:
    def __init__(self, p, amplitude, random_state=2022):
        np.random.seed(random_state)
        self.p = p
        self.a = amplitude

    def _sample_clean(self, n):
        return None

    def _sample_outlier(self, n):
        return None

    def sample(self, n, purity, random_state=2022):
        p = self.p
        np.random.seed(random_state)
        purity = np.clip(purity, 0, 1)
        n_clean = np.round(n * purity).astype(int)
        n_outlier = n - n_clean
        X_clean = self._sample_clean(n_clean)
        is_outlier = np.zeros((n,))
        if n_outlier > 0:
            X_outlier = self._sample_outlier(n_outlier)
            idx_clean, idx_outlier = train_test_split(np.arange(n), test_size=n_outlier)
            X = np.zeros((n,p))
            X[idx_clean,:] = X_clean
            X[idx_outlier,:] = X_outlier
            is_outlier[idx_outlier] = 1
        else:
            X = X_clean
        return X, is_outlier.astype(int)


class GaussianMixture(DataModel):
    def _sample_clean(self, n):
        p = self.p
        X = np.random.randn(n, p)
        return X

    def _sample_outlier(self, n):
        p = self.p
        X = self.a + np.random.randn(n, p)
        return X


class ConcentricCircles(DataModel):
    def _sample_clean(self, n):
        p = self.p
        X = np.random.randn(n, p)
        return X

    def _sample_outlier(self, n):
        p = self.p
        X = np.sqrt(self.a)*np.random.randn(n, p)
        return X

class ConcentricCirclesMixture(DataModel):
    "This seems unecessarily complicated"
    def __init__(self, p, amplitude, random_state=2022):
        super().__init__(p, amplitude, random_state=random_state)
        self.Z = np.random.uniform(low=-3, high=3, size=(p,p))

    def _sample_clean(self, n):
        p = self.p
        X = np.random.randn(n, p)
        cluster_idx = np.random.choice(self.Z.shape[0], n, replace=True)
        X = X + self.Z[cluster_idx,]
        return X

    def _sample_outlier(self, n):
        p = self.p
        X = np.sqrt(self.a) * np.random.randn(n, p)
        cluster_idx = np.random.choice(self.Z.shape[0], n, replace=True)
        X = X + self.Z[cluster_idx,]
        return X


class BinomialModel(DataModel):
    def __init__(self, p, amplitude, random_state=2022):
        super().__init__(p, amplitude, random_state=random_state)
        self.beta_Z = np.sqrt(amplitude)*np.random.normal(size=(p,2))

    def sample(self, n, purity=1, random_state=2022):

        def calculate_offset(purity):
            X = sample_X(1000)
            def foo(offset):
                Y = sample_Y(X, offset)
                return np.mean(Y) - (1.0-purity)
            offset = optimize.bisect(foo, -1000, 1000)
            return offset
 
        def sample_X(n):
            X = np.random.normal(0, 1, (n,self.p))
            X[:,0] = np.random.uniform(low=0, high=1, size=(n,))
            factor = 0.1
            idx_1 = np.where(X[:,0]<=factor)[0]
            idx_2 = np.where(X[:,0]>factor)[0]
            X[idx_1,0] = 0
            X[idx_2,0] = 0
            return X

        def compute_prob(X, offset):
            f = np.matmul(X,self.beta_Z)
            f[:,0] = f[:,0] - offset/2
            f[:,1] = f[:,1] + offset/2
            prob = np.exp(f)
            prob_y = prob / np.expand_dims(np.sum(prob,1),1)
            return prob_y

        def sample_Y(X, offset):
            prob_y = compute_prob(X, offset)
            g = np.array([np.random.multinomial(1,prob_y[i]) for i in range(X.shape[0])], dtype = float)
            classes_id = np.arange(2)
            y = np.array([np.dot(g[i],classes_id) for i in range(X.shape[0])], dtype = int)
            return y

        offset = calculate_offset(purity)
        X = sample_X(n)
        is_outlier = sample_Y(X, offset)
        return X, is_outlier.astype(int)
