import numpy as np
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from scipy.io import arff


def load_outlier_data(base_path, filename):
    if filename.endswith('.csv'):
        data_raw = pd.pandas.read_csv(base_path + filename)
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
        
        
    return data_raw

def get_data_is_outlier(name, base_path):
    
    data_raw = load_outlier_data(base_path, name)
    p = data_raw.shape[1]
    
    # Extract outliers
    is_outlier = (data_raw['Class']==1)
    data = np.array(data_raw)[:,0:(p-1)]
    
    shuffle = np.random.permutation(data.shape[0])
    data = data[shuffle]
    is_outlier = is_outlier[shuffle]
    
    return data, np.array(is_outlier.astype(int))


