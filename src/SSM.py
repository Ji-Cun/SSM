import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
import time
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tensorflow import keras

from src.DataIO import loadDataFromTsv,loadTrainTranDataFromTsv
from src.Segment import getSeriesFeatures
from src.classifiers.Classifier import FCN

def generateCandidates(x_train):
    # generate cadidates
    features = []
    # 最大长度为100或者时间序列长度、最小长度设为10，期望的分段数为长度除以10和3之间的较大值
    maxLength = min(len(x_train[0]), 100)
    minLength = min(math.ceil(len(x_train[0]) / 5), 10)
    segNumber = max(math.ceil(len(x_train[0]) / 20), 5)
    for seriesId in range(len(x_train)):
        values = x_train[seriesId]
        seriesFeatures = getSeriesFeatures(seriesId, values, segNumber, maxLength, minLength)
        for feature in seriesFeatures:
            features.append(feature)
    # print('candidate number', len(features))

    return features


'''
def transform(features, dataset):
    instanceNum = len(dataset)
    featureNum = len(features)
    distTrans = np.zeros((instanceNum, featureNum))
    postionTrans = np.zeros((instanceNum, featureNum))
    for j in range(featureNum):
        if j % 50 == 0:
            print('转换进度:', j, '/', featureNum)
        for i in range(instanceNum):
            distTrans[i, j], postionTrans[i, j] = getMinDistance(features[j], dataset[i])
    return distTrans, postionTrans
'''


def transform2(features, dataset):
    instanceNum = len(dataset)
    featureNum = len(features)
    distTrans = np.zeros((instanceNum, featureNum))
    postionTrans = np.zeros((instanceNum, featureNum))
    for i in range(instanceNum):
        parResult = Parallel(n_jobs=-1)(delayed(getMinDistance)(feature, dataset[i]) for feature in features)
        for j in range(featureNum):
            distTrans[i, j], postionTrans[i, j] = parResult[j]
    return distTrans, postionTrans


def transform(features, dataset):
    instanceNum = len(dataset)
    featureNum = len(features)
    distTrans = np.zeros((instanceNum, featureNum))
    postionTrans = np.zeros((instanceNum, featureNum))

    parResult = Parallel(n_jobs=-1)(delayed(getMinDistance)(feature, timeseries) for timeseries in dataset for feature in features)

    for i in range(instanceNum):
        for j in range(featureNum):
            distTrans[i, j], postionTrans[i, j] = parResult[i*featureNum+j]
    return distTrans, postionTrans



def getMinDistance(feature, timeSeries):
    min = sys.float_info.max
    minPostion = 0
    subsequence=feature.values
    l1 = len(subsequence)
    l2 = len(timeSeries)
    for i in range(0, l2 - l1 + 1):
        dist = 0.0
        for j in range(0, l1):
            dist = dist + pow(subsequence[j] - timeSeries[i + j], 2)
            if dist >= min:
                break
        if dist < min:
            min = dist
            minPostion = i
    return np.sqrt(min / l1), minPostion

def featureSelection(train_dist, train_postion, y_train_origin, feature_number, candidates, evalution_method='RF'):
    #分类器设置
    begin = time.time()
    if evalution_method=='RF':
        classifier = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        classifier.fit(train_dist, y_train_origin)
    elif evalution_method=='TREE':
        classifier = DecisionTreeClassifier(random_state=0)
        classifier.fit(train_dist, y_train_origin)
    elif evalution_method=='ET':
        classifier = ExtraTreesClassifier(n_estimators=200,random_state=0)
        classifier.fit(train_dist, y_train_origin)
    elif evalution_method=='ADA':
        classifier = AdaBoostClassifier(n_estimators=200,random_state=0)
        classifier.fit(train_dist, y_train_origin)
    else:
        classifier = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        classifier.fit(train_dist, y_train_origin)
    '''特征选择'''
    importances = classifier.feature_importances_
    end = time.time()
    evalution_time = end - begin


    indices = np.argsort(importances)[::-1]
    selectedIndex = []
    for i in range(len(indices)):
        index1 = indices[i]
        if importances[index1] > 0:
            filtered =False
            for index2 in selectedIndex:
                postion1 = train_postion[:, index1]
                postion2 = train_postion[:, index2]

                count = 0
                for k in range(len(postion1)):
                    intersection = 0

                    if (postion1[k] <= postion2[k] and postion1[k] + candidates[index1].length > postion2[k]):
                        intersection = min(postion1[k] + candidates[index1].length - postion2[k],
                                           candidates[index2].length)
                    if (postion2[k] <= postion1[k] and postion2[k] + candidates[index2].length > postion1[k]):
                        intersection = min(postion2[k] + candidates[index2].length - postion1[k],
                                           candidates[index1].length)

                    if intersection / candidates[index1].length >= 0.9 and intersection / candidates[
                        index2].length >= 0.9:
                        count = count + 1
                if count / len(postion1) >= 0.9:
                    filtered = True
                    break

            if filtered==False:
                selectedIndex.append(index1)

            if len(selectedIndex) >= feature_number:
                break
        else:
            break
    selectedIndex = np.array(selectedIndex)
    selectedIndex.sort()

    features = []
    for index in selectedIndex:
        feature = candidates[index]
        features.append(feature)

    return features, selectedIndex, evalution_time


def pairwiseRelativePostion(postion):
    prp = np.zeros((len(postion), len(postion[0]), len(postion[0])))
    for n in range(len(postion)):
        for i in range(len(postion[0])):
            for j in range(len(postion[0])):
                prp[n][i][j] = abs(postion[n][i] - postion[n][j])
    return prp


def accuracy(dataset='Beef', feature_number='256',evalution_method='RF'):
    # load data
    x_train_origin, y_train_origin, x_test_origin, y_test_origin = loadDataFromTsv(dataset)

    # generate cadidates
    candidates = generateCandidates(x_train_origin)

    if os.path.exists('..\\trans\\dist\\' + dataset + '_TRAIN.tsv') and os.path.exists('..\\trans\\position\\' + dataset + '_TRAIN.tsv') :
        train_dist = loadTrainTranDataFromTsv(dataset,type='dist')
        train_postion = loadTrainTranDataFromTsv(dataset,type='position')

    else:
        # transformation
        if dataset=='StarLightCurves':
            train_dist, train_postion = transform2(candidates, x_train_origin)
        else:
            train_dist, train_postion = transform(candidates, x_train_origin)

        y_reshape=y_train_origin.reshape(-1, 1)
        np.savetxt('..\\trans\\dist\\' + dataset + '_TRAIN.tsv', np.append(y_reshape,train_dist,axis=1), delimiter='\t')
        np.savetxt('..\\trans\\position\\' + dataset + '_TRAIN.tsv', np.append(y_reshape, train_postion,axis=1), delimiter='\t')

    '''特征选择'''
    features, selectedIndex,evalution_time = featureSelection(train_dist, train_postion, y_train_origin, feature_number, candidates, evalution_method=evalution_method)

    '''降维'''
    train_dist = train_dist[:, selectedIndex]
    test_dist, _ = transform(features, x_test_origin)


    '''categorical Y'''
    nb_classes = len(np.unique(y_train_origin))
    y_train = np.round((y_train_origin - y_train_origin.min()) / (y_train_origin.max() - y_train_origin.min()) * (
            nb_classes - 1)).astype(np.int32)
    y_test = np.round((y_test_origin - y_train_origin.min()) / (y_train_origin.max() - y_train_origin.min()) * (nb_classes - 1)).astype(np.int32)
    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.to_categorical(y_test, nb_classes)

    ''' process dist as input1 '''
    dist_mean = train_dist.mean()
    dist_std = train_dist.std()
    train_dist = (train_dist - dist_mean) / (dist_std)
    test_dist = (test_dist - dist_mean) / (dist_std)
    # add a dimension to make it multivariate with one dimension
    distTrain = train_dist.reshape((train_dist.shape[0], train_dist.shape[1], 1))
    distTest = test_dist.reshape((test_dist.shape[0], test_dist.shape[1], 1))

    fcn = FCN(distTrain.shape[1:], nb_classes)
    acc = fcn.fit(distTrain, distTest, Y_train, Y_test)

    keras.backend.clear_session()

    return acc,evalution_time




def diversity(dataset='Beef', feature_number='256'):
    # load data
    x_train_origin, y_train_origin, x_test_origin, y_test_origin = loadDataFromTsv(dataset)

    # generate cadidates
    candidates = generateCandidates(x_train_origin)

    if os.path.exists('..\\trans\\dist\\' + dataset + '_TRAIN.tsv') and os.path.exists('..\\trans\\position\\' + dataset + '_TRAIN.tsv') :
        train_dist = loadTrainTranDataFromTsv(dataset,type='dist')
        train_postion = loadTrainTranDataFromTsv(dataset,type='position')

    else:
        # transformation
        if dataset=='StarLightCurves':
            train_dist, train_postion = transform2(candidates, x_train_origin)
        else:
            train_dist, train_postion = transform(candidates, x_train_origin)

        y_reshape=y_train_origin.reshape(-1, 1)
        np.savetxt('..\\trans\\dist\\' + dataset + '_TRAIN.tsv', np.append(y_reshape,train_dist,axis=1), delimiter='\t')
        np.savetxt('..\\trans\\position\\' + dataset + '_TRAIN.tsv', np.append(y_reshape, train_postion,axis=1), delimiter='\t')


    diversityDraw(train_dist, train_postion, y_train_origin, feature_number, candidates)

    return 0

def diversityDraw(train_dist, train_postion, y_train_origin, feature_number, candidates):
    '''特征选择'''
    classifier = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    classifier.fit(train_dist, y_train_origin)
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    filtered = set()
    selectedIndex = []
    for i in range(len(indices)):
        index1 = indices[i]
        if importances[index1] > 0:
            if index1 in filtered:
                continue
            else:
                selectedIndex.append(index1)
                if len(selectedIndex) >= feature_number:
                    break
                for j in range(i + 1, len(indices)):
                    index2 = indices[j]
                    # 判断是否要加入到filter
                    postion1 = train_postion[:, index1]
                    postion2 = train_postion[:, index2]

                    count = 0
                    for k in range(len(postion1)):
                        intersection = 0

                        if (postion1[k] <= postion2[k] and postion1[k] + candidates[index1].length > postion2[k]):
                            intersection = min(postion1[k] + candidates[index1].length - postion2[k],
                                               candidates[index2].length)
                        if (postion2[k] <= postion1[k] and postion2[k] + candidates[index2].length > postion1[k]):
                            intersection = min(postion2[k] + candidates[index2].length - postion1[k],
                                               candidates[index1].length)

                        if intersection / candidates[index1].length >= 0.9 and intersection / candidates[
                            index2].length >= 0.9:
                            count = count + 1
                    if count / len(postion1) >= 0.9:
                        filtered.add(index2)
        else:
            break

    selectedIndex = np.array(selectedIndex)

    num=len(selectedIndex)

    fig = plt.figure(figsize=(7, 8))
    for i in range(num):
        plt.subplots_adjust(hspace=2)
        ax1 = fig.add_subplot(num, 2, (2*i+1))  # 参数为 (rows, columns, index)
        title = 'Shapelet '+str(i+1)+' of SSM-R'
        ax1.set_title(title)
        print(indices[i])
        ax1.plot(candidates[indices[i]])

        ax1 = fig.add_subplot(num, 2, (2*i + 2))  # 参数为 (rows, columns, index)
        title = 'Shapelet '+str(i+1)+' of SSM'
        ax1.set_title(title)
        ax1.plot(candidates[selectedIndex[i]])
        print(selectedIndex[i])
    plt.show()




