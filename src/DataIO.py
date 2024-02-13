import pandas as pd
import numpy as np

file_path = 'G:\\time series data\\UCRArchive_2018\\'


def loadDataFromTsv(dataset):
    train_path = file_path + dataset + '\\' + dataset + '_TRAIN.tsv'
    test_path = file_path + dataset + '\\' + dataset + '_TEST.tsv'

    df_train = pd.read_csv(train_path, sep='\t', header=None)
    df_test = pd.read_csv(test_path, sep='\t', header=None)

    y_train = df_train.values[:, 0].astype(np.int32)
    y_test = df_test.values[:, 0].astype(np.int32)

    x_train = df_train.drop(columns=[0])
    x_test = df_test.drop(columns=[0])

    x_train.columns = range(x_train.shape[1])
    x_test.columns = range(x_test.shape[1])

    x_train = x_train.values
    x_test = x_test.values

    # znorm

    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    return x_train, y_train, x_test, y_test


def loadTrainTranDataFromTsv(dataset,type='dist'):
    if type=='dist':
        train_path='..\\trans\\dist\\'+dataset+'_TRAIN.tsv'
    elif type=='position':
        train_path = '..\\trans\\position\\' + dataset + '_TRAIN.tsv'

    if dataset=='StarLightCurves':
        df_train = pd.read_csv(train_path, sep='\t', header=None,low_memory=False)
    else:
        df_train = pd.read_csv(train_path, sep='\t', header=None)
    y_train = df_train.values[:, 0].astype(np.int32)
    x_train = df_train.drop(columns=[0])
    x_train.columns = range(x_train.shape[1])
    x_train = x_train.values
    return x_train