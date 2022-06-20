import numpy as np
import statistics as stats
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

class Model_CC:
    def __init__(self, p, amplitude, random_state=2020):
        assert amplitude >= 1
        #np.random.seed(random_state)
        self.p = p
        self.a = amplitude
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

    def sample(self, n, purity=1, random_state=2020):
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


class Model_Binomial:
    def __init__(self, p, amplifier, K=2, magnitude=1):
        self.K = K
        self.p = p
        self.magnitude = magnitude
        self.amplifier = amplifier
        # Generate model parameters
        self.beta_Z = self.magnitude*np.random.randn(self.p,self.K)
        #self.beta_Z[0,0] = 1
        #self.beta_Z[0,1] = 0.5

    def sample_X(self, n):
        X = np.random.normal(0, 1, (n,self.p))
        factor = 0
        X[0:int(n*factor),0] = -1
        X[int(n*factor):,0] = 8
        #X[:,0] = self.amplifier
        return X.astype(np.float32)
    
    def compute_prob(self, X):
        f = np.matmul(X,self.beta_Z)
        prob = np.exp(f)
        prob_y = prob / np.expand_dims(np.sum(prob,1),1)
        return prob_y

    def sample_Y(self, X):
        prob_y = self.compute_prob(X)
        g = np.array([np.random.multinomial(1,prob_y[i]) for i in range(X.shape[0])], dtype = float)
        classes_id = np.arange(self.K)
        y = np.array([np.dot(g[i],classes_id) for i in range(X.shape[0])], dtype = int)
        return y.astype(np.int)
    
    def sample(self, n, purity=1, random_state=2020):
        p = self.p
        purity = np.clip(purity, 0, 1)
        n_clean = np.round(n * purity).astype(int)
        n_outlier = n - n_clean
        X_clean = np.array([[]]).reshape(0,p)
        X_outlier = np.array([[]]).reshape(0,p)
        
        while (np.shape(X_clean)[0] < n_clean) or (np.shape(X_outlier)[0] < n_outlier):
          X_data = self.sample_X(2*n)
          Y_data = self.sample_Y(X_data)
          X_clean = np.concatenate((X_clean, X_data[np.where(Y_data==0)[0]]), axis=0)
          X_outlier = np.concatenate((X_outlier, X_data[np.where(Y_data==1)[0]]), axis=0)
        
        X_clean = X_clean[:n_clean,]
        X_outlier = X_outlier[:n_outlier,]
        is_outlier = np.zeros((n,))

        if n_outlier > 0:
            idx_clean, idx_outlier = train_test_split(np.arange(n), test_size=n_outlier)
            X = np.zeros((n,p))
            X[idx_clean,:] = X_clean
            X[idx_outlier,:] = X_outlier
            is_outlier[idx_outlier] = 1
        else:
            X = X_clean
        return X, is_outlier.astype(int)


## add random state if needed
class Model_Gaussian:
  def __init__(self, p, sep, random_state=2022):
    np.random.seed(random_state)
    self.p = p         
    self.sep = sep            # determines the seperation of two gaussian blobs
    self.center = np.empty((2,p))
    self.center[0] = np.random.uniform(low=-3,high=3, size=p)
    self.center[1] = self.center[0] + sep

  def sample(self, n, purity=1, random_state=2022):
    p = self.p
    sep = self.sep
    np.random.seed(random_state)
    purity = np.clip(purity, 0, 1)
    n_clean = np.round(n * purity).astype(int)
    n_outlier = n - n_clean
    X, Y = make_blobs(n_samples=[n_clean,n_outlier], centers = self.center, 
                      n_features = p, random_state=random_state)
    return X, Y