import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, fetch_covtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import re
import pickle
import pdb

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

    def __init__(self, base_path, data_name, random_state=None):        
        # Load the data
        if data_name=="digits":
            dataraw = load_digits()
            X = dataraw['data']
            Y = dataraw['target'] == 0
        elif data_name=="covtype":
            dataraw = fetch_covtype()
            X = dataraw['data']
            Y = dataraw['target'] == 1
        elif data_name=="toxicity":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=";", header=None)
            Y = (np.array(data_raw.iloc[:,-1])=='positive').astype(int)
            X = np.array(data_raw.iloc[:,:-1])
        elif data_name=="ad":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None, na_values=['?','     ?','   ?'])
            data_raw = data_raw.fillna(data_raw.median())
            Y = (np.array(data_raw.iloc[:,-1])=='ad.').astype(int)
            X = np.array(data_raw.iloc[:,:-1])
        elif data_name=="androgen":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=";", header=None)
            Y = (np.array(data_raw.iloc[:,-1])=='positive').astype(int)
            X = np.array(data_raw.iloc[:,:-1])
        elif data_name=="rejafada":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None).iloc[:,1:]
            Y = (np.array(data_raw.iloc[:,0])=='M').astype(int)
            X = np.array(data_raw.iloc[:,1:])
        elif data_name=="hepatitis":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None, na_values=['?','     ?','   ?'])
            data_raw = data_raw.fillna(data_raw.median())
            Y = (np.array(data_raw.iloc[:,0])==1).astype(int)
            X = np.array(data_raw.iloc[:,1:])
        elif data_name=="ctg":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",", header=None, na_values=['?','     ?','   ?'])
            data_raw = data_raw.fillna(data_raw.median())
            Y = (np.array(data_raw.iloc[:,-1])==2).astype(int)
            X = np.array(data_raw.iloc[:,:-1])
        elif data_name=="creditcard":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",")
            data_raw = data_raw.fillna(data_raw.median())
            Y = (np.array(data_raw.iloc[:,-1])==1).astype(int)
            X = np.array(data_raw.iloc[:,:-1])
        elif data_name=="seizures":
            data_raw = pd.pandas.read_csv(base_path + data_name + ".csv", sep=",")
            idx_keep = np.where(data_raw.iloc[:,-1]!=1)[0]
            Y = (np.array(data_raw.iloc[idx_keep,-1])==5).astype(int)
            X = np.array(data_raw.iloc[idx_keep,1:-1])
        elif data_name=="splice":
            base_path = "../experiments_real/data/"
            data_raw = pd.pandas.read_csv(base_path + "splice.data", header=None, sep=",")
            X_raw = data_raw.iloc[:,-1]

            label_encoder = LabelEncoder()
            label_encoder.fit(np.array(['a','c','g','t','n']))

            def string_to_array(seq_string):
                seq_string = seq_string.lower().strip()
                seq_string = re.sub('[^acgt]', 'n', seq_string)
                seq_string = np.array(list(seq_string))
                return seq_string

            int_encoded = label_encoder.transform(string_to_array('acgtn'))
            onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
            int_encoded = int_encoded.reshape(len(int_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(int_encoded)

            def one_hot_encoder(seq_string):
                int_encoded = label_encoder.transform(seq_string)
                int_encoded = int_encoded.reshape(len(int_encoded), 1)
                onehot_encoded = onehot_encoder.transform(int_encoded)
                onehot_encoded = np.delete(onehot_encoded, -1, 1)
                return onehot_encoded

            X = [ one_hot_encoder(string_to_array(x)).flatten() for x in X_raw]
            X = np.stack(X)
            Y = np.array(data_raw.iloc[:,0]=="IE").astype(int)

        elif data_name=="cifar-100":
            def unpickle(file):
                import pickle
                with open(file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                return dict

            base_path = "../experiments_real/data/"
            metadata_path = base_path + 'cifar-100/meta' # change this path`\
            metadata = unpickle(metadata_path)
            superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))

            data_pre_path = base_path + 'cifar-100/' # change this path
            # File paths
            data_train_path = data_pre_path + 'train'
            data_test_path = data_pre_path + 'test'
            # Read dictionary
            data_train_dict = unpickle(data_train_path)
            data_test_dict = unpickle(data_test_path)
            # Get data (change the coarse_labels if you want to use the 100 classes)
            X = data_train_dict[b'data']
            Y = np.array(data_train_dict[b'coarse_labels'])
            idx_keep = np.arange(len(Y)) #np.where((Y==18)+(Y==19)+(Y==5)+(Y==6)+(Y==14)>0)[0]
            X = X[idx_keep]
            Y = Y[idx_keep]
            X = X.astype(float)
            Y = (Y!=14).astype(int)
        
        elif data_name=="cifar-10":
            cifar_10_dir = "../experiments_real/data/cifar-10"
            X, _, Y, test_data, _, _ = load_cifar_10_data(cifar_10_dir)
            idx_keep = np.where((Y==0)+(Y==3)+(Y==4)+(Y==5)+(Y==6)+(Y==7)>0)[0]
            X = X[idx_keep]
            Y = (Y[idx_keep] == 0).astype(int)
            
        else:
            X, Y = self._load_outlier_data(base_path, data_name + ".mat")
        print("Loaded data set with {:d} samples: {:d} inliers, {:d} outliers.".format(len(Y), np.sum(Y==0), np.sum(Y==1)))

        # Extract test set
        if random_state is not None:
            np.random.seed(random_state)            
        idx_in = np.where(Y==0)[0]
        idx_out = np.where(Y==1)[0]
        idx_test_out = np.random.choice(idx_out, int(len(idx_out)/5), replace=False)
        idx_test_in = np.random.choice(idx_in, np.minimum(len(idx_in), len(idx_test_out)), replace=False)
        idx_test = np.append(idx_test_out, idx_test_in)
        idx_train = np.setdiff1d(np.arange(len(Y)), idx_test)
        np.random.shuffle(idx_train)
        np.random.shuffle(idx_test)
        self.X = X[idx_train]
        self.Y = Y[idx_train]
        self.X_test = X[idx_test]
        self.Y_test = Y[idx_test]
        self.n_out = np.sum(self.Y==1)
#        pdb.set_trace()

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

    def sample_test(self, n=None, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        if n is None:
            idx_sample = np.arange(len(self.Y_test))
        else:
            idx_sample = np.random.choice(len(self.Y_test), n)
        return self.X_test[idx_sample], self.Y_test[idx_sample]

    def sample(self, n_in=None, n_out=None, mislabeled_prop=0, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)    
        if n_out is None:
            n_out = np.sum(self.Y==1)
        if n_in is None:
            n_in = np.sum(self.Y==0)
        
        idx_in = np.where(self.Y==0)[0]
        idx_out = np.where(self.Y==1)[0]
        n_out = np.minimum(n_out, len(idx_out))
        n_in = np.minimum(n_in, len(idx_in))

        idx_sample_out = np.random.choice(idx_out, n_out, replace=False)
        idx_sample_in = np.random.choice(idx_in, n_in, replace=False)
        Y = self.Y
        
        if mislabeled_prop>0:
            n_out_mis = np.minimum(n_in, int(n_out*mislabeled_prop))
            idx_sample_in, idx_sample_in_out = train_test_split(idx_sample_in, test_size=n_out_mis, random_state=random_state)
            idx_sample_out = np.append(idx_sample_out, idx_sample_in_out)
            Y[idx_sample_in_out] = 1
        
        idx_sample = np.append(idx_sample_in, idx_sample_out)
        np.random.shuffle(idx_sample)
        
        return self.X[idx_sample], Y[idx_sample]
