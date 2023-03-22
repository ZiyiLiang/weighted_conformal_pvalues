import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import mat73
import re
import pickle
from scipy.io import arff, loadmat
import pdb
from joblib import Memory
memory = Memory('./tmp')
fetch_openml_cached = memory.cache(fetch_openml)

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    #cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    #if negatives:
    #    cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    #else:
    #    cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels

class DataSet:

    def __init__(self, base_path, data_name, random_state=None, outlier_shift=0):
        self.outlier_shift = outlier_shift
        self.outlier_train_group = None

        # Load the data
        if data_name=="images_flowers":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None)
            Y = np.array(data_raw.iloc[:,0])
            X = np.array(data_raw.iloc[:,1:])
            labels_inlier = ["roses"]
            labels_outlier_train = []
            labels_outlier_test = ["sunflowers","dandelion","daisy","tulips"]
            label_groups = np.zeros((len(labels_outlier_test,)))

        elif data_name=="images_animals":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None)
            Y = np.array(data_raw.iloc[:,0])
            X = np.array(data_raw.iloc[:,1:])
            labels_inlier = ["hamster", "guinea pig"]
            labels_outlier_train = []
            labels_outlier_test = ["lynx","wolf","coyote","cheetah","jaguer","chimpanzee","orangutan","cat"]
            label_groups_dict = {"hamster":0, "guinea pig":0, "lynx":1,"wolf":2,"coyote":2,"cheetah":1,"jaguer":1,"chimpanzee":3,"orangutan":3,"cat":1}
            self.label_test_group = 2

        elif data_name=="images_cars":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None)
            Y = np.array(data_raw.iloc[:,0])
            X = np.array(data_raw.iloc[:,1:])
            labels_inlier = ["car"]
            labels_outlier_train = []
            labels_outlier_test = ["fruit", "dog", "motorbike", "person", "cat", "flower", "airplane"]
            label_groups = np.zeros((len(labels_outlier_test,)))

        elif data_name=="mammography":
            mat = loadmat(base_path + "mammography.mat")
            X = mat['X']
            Y = mat['y']
            labels_inlier = [0]
            labels_outlier_train = []
            labels_outlier_test = [1]
            label_groups = np.zeros((len(labels_outlier_test,)))

        elif data_name=="annthyroid":
            mat = loadmat(base_path + "annthyroid.mat")
            X = mat['X']
            Y = mat['y'].flatten()
            labels_inlier = [0]
            labels_outlier_train = []
            labels_outlier_test = [1]
            label_groups = np.zeros((len(labels_outlier_test,)))

        is_inlier = np.array([y in labels_inlier for y in Y]).astype(int)
        is_outlier = 1-is_inlier
        is_outlier_train = np.array([y in labels_outlier_train for y in Y]).astype(int)
        is_outlier_test = np.array([y in labels_outlier_test for y in Y]).astype(int)
        label_groups = np.array([label_groups_dict[y] for y in Y]).astype(int)

        print("Loaded data set with {:d} samples: {:d} inliers and {:d} outliers, of which {:d} are available for training."\
              .format(len(Y), np.sum(is_inlier), np.sum(is_outlier), np.sum(is_outlier_train)))

        # Define test set
        if random_state is not None:
            np.random.seed(random_state)

        # Define lists of inliers and outliers
        self.idx_in = np.where((is_outlier==0))[0]
        self.idx_out = np.where((is_outlier==1))[0]

        self.X = X
        self.Y = is_outlier
        self.Y_label = Y
        self.label_groups = label_groups
        self.n_in = np.sum(self.Y==0)
        self.n_out = np.sum(self.Y==1)

        # self.X_test = X[idx_test]
        # self.Y_test = is_outlier[idx_test]
        # self.n_in = np.sum(self.Y==0)
        # self.n_out = np.sum(self.Y==1)
        # self.label_groups = label_groups[idx_train]
        # self.label_groups_test = label_groups[idx_test]   

    def reset(self):
        # Re-define lists of available inliers and outliers
        self.idx_in = np.where((self.Y==0))[0]
        self.idx_out = np.where((self.Y==1))[0]


    def sample_test(self, n, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
      
        # List of available inliers and outliers
        idx_out = self.idx_out
        idx_in = self.idx_in

        # Sample the outliers
        if self.outlier_shift>0:
            self.label_test_group = 2
            groups = self.label_groups[idx_out]
            is_target = np.zeros((len(idx_out),))
            idx_target = np.where(groups==self.label_test_group)[0]
            is_target[idx_target] = 1
            p_test_out = (1.0-self.outlier_shift) * np.ones((len(idx_out),)) + self.outlier_shift * is_target
            p_test_out = p_test_out / np.sum(p_test_out) 
            sample_out = np.random.choice(idx_out, n, replace=False, p=p_test_out)      
        else:
            sample_out = np.random.choice(idx_out, n, replace=False)      

        # Sample the inliers
        sample_in = np.random.choice(idx_in, n, replace=False)             

        # Debug
        print("Sampled outliers in test set:")
        print(np.array(np.unique(self.Y_label[sample_out], return_counts=True)).T)
        print("Sampled inliers in test set:")
        print(np.array(np.unique(self.Y_label[sample_in], return_counts=True)).T)

        # Re-define lists of available inliers and outliers
        self.idx_in = np.setdiff1d(self.idx_in, sample_in)
        self.idx_out = np.setdiff1d(self.idx_out, sample_out)

        # Output the sampled inliers and outliers
        idx_sample = np.concatenate((sample_in,sample_out))
        np.random.shuffle(idx_sample)

        return self.X[idx_sample], self.Y[idx_sample]

        
    def sample(self, n_in=None, n_out=None, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
      
        # List of available inliers and outliers
        idx_out = self.idx_out
        idx_in = self.idx_in

        if n_out is None:
            n_out = len(self.idx_out)
        if n_in is None:
            n_in = len(self.idx_in)

        # Sample the outliers
        if self.outlier_shift>0:
            self.label_test_group = 2
            groups = self.label_groups[idx_out]
            is_target = np.zeros((len(idx_out),))
            idx_target = np.where(groups==self.label_test_group)[0]
            is_target[idx_target] = 1
            p_test_out = np.ones((len(idx_out),)) - self.outlier_shift * is_target
            p_test_out = p_test_out / np.sum(p_test_out) 
            sample_out = np.random.choice(idx_out, n_out, replace=False, p=p_test_out)      
        else:
            sample_out = np.random.choice(idx_out, n_out, replace=False)      

        # Sample the inliers
        sample_in = np.random.choice(idx_in, n_in, replace=False)             

        # Debug
        print("Sampled outliers in train set:")
        print(np.array(np.unique(self.Y_label[sample_out], return_counts=True)).T)
        print("Sampled inliers in train set:")
        print(np.array(np.unique(self.Y_label[sample_in], return_counts=True)).T)

        # Re-define lists of available inliers and outliers
        self.idx_in = np.setdiff1d(self.idx_in, sample_in)
        self.idx_out = np.setdiff1d(self.idx_out, sample_out)

        # Output the sampled inliers and outliers
        idx_sample = np.concatenate((sample_in,sample_out))
        np.random.shuffle(idx_sample)

        return self.X[idx_sample], self.Y[idx_sample]

    def _load_outlier_data(self, base_path, filename, sep=","):
        if filename.endswith('.csv'):
            data_raw = pd.pandas.read_csv(base_path + filename, sep=sep)
            pdb.set_trace()
        elif filename.endswith('.mat'):
            if 'http' in filename:
                mat = mat73.loadmat(base_path + filename)
            else:
                mat = loadmat(base_path + filename)
            X = mat['X']
            Y = mat['y'].reshape(-1,1)
            data = np.concatenate((X,Y),axis=1)
            index   = [str(i) for i in range(0, len(Y))]
            columns = ["X_" + str(i) for i in range(0, X.shape[1])]
            columns.append("Class")
            data_raw = pd.DataFrame(data=data,index=index,columns=columns)
        elif filename.endswith('.arff'):
            data = arff.loadarff(base_path + filename)
            data_raw = pd.DataFrame(data[0])
            data_raw = data_raw.drop(columns=['id'])
            data_raw = data_raw.rename(columns={"outlier": "Class"})
            data_raw['Class'] = (data_raw['Class']==b'yes').astype(float)

        X = np.array(data_raw.drop("Class", axis=1))
        p = X.shape[1]
        Y = np.array(data_raw["Class"]).astype(int)
        return X, Y
