#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:24:00 2018

@author: charlesdickens
"""

from __future__ import division
from fancyimpute import KNN as KNNImpute
import csv    
import pymonad
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
import re
import math

def binarizeCategoricalData(X, data):
    mlb = MultiLabelBinarizer()
    
    # male or female
    X = [list(mlb.fit_transform([[d[1]] for d in data[1:]])[i]) + X[i][1:] for i in range(len(X))]
    
    # patient had cancer
    X = [X[i][0:4] + list(mlb.fit_transform([[d[4]] for d in data[1:]])[i]) + X[i][5:] for i in range(len(X))]
    
    # patient had chemo
    X = [X[i][0:6] + list(mlb.fit_transform([[d[5]] for d in data[1:]])[i]) + X[i][7:] for i in range(len(X))]
    
    # patient had radiation
    X = [X[i][0:8] + list(mlb.fit_transform([[d[6]] for d in data[1:]])[i]) + X[i][9:] for i in range(len(X))]
    
    # whether the patient was diagnosed with an infection
    X = [X[i][0:10] + list(mlb.fit_transform([[d[7]] for d in data[1:]])[i]) + X[i][11:] for i in range(len(X))]
    
    # the cytogenic category of the patient
    X = [X[i][0:12] + list(mlb.fit_transform([[d[8]] for d in data[1:]])[i]) + X[i][13:] for i in range(len(X))]
    
    # whether the patient was found to have a ITD FLT3 mutation
    X = [X[i][0:26] + list(mlb.fit_transform([[d[9]] for d in data[1:]])[i]) + X[i][27:] for i in range(len(X))]
    
    # whether the patient was found to have a D835 FLT3 mutation
    X = [X[i][0:29] + list(mlb.fit_transform([[d[10]] for d in data[1:]])[i]) + X[i][30:] for i in range(len(X))]
    
    # whether the patient was found to have a Ras.Stat mutation
    X = [X[i][0:32] + list(mlb.fit_transform([[d[11]] for d in data[1:]])[i]) + X[i][33:] for i in range(len(X))]
    
    # the specific anthra based treatment adminstered
    X = [X[i][0:35] + list(mlb.fit_transform([[d[12]] for d in data[1:]])[i]) + X[i][36:] for i in range(len(X))]
    
    return X


"""
function: crossValidation(X, Y, k, L, Method)

parameters:  X: 
             Y: 
             k:
             Method:
             Q:
    
description: 
"""
def crossValidation(X, Y, k, Method, Q, preprocess):
    predictionErrors = []
    size = int(math.floor(len(Y) / k))
    
    for i in range(k):
        # break up data into validation(test) and training sets
        Xtest = X[i*size:i*size + size]
        Ytest = Y[i*size:i*size + size]
        Xtraining = [X[l] for l in range(len(X)) if l < i*size or l > i*size + size - 1]
        Ytraining = [Y[l] for l in range(len(Y)) if l < i*size or l > i*size + size - 1]
        
        # Method will train the model and return a method to make predictions
        Predictor = Method(Xtraining, Ytraining, preprocess)
        
        # make predictions on the test set and calculate the prediction error
        # using the Q function passed to this function
        predictionErrors.append(Q(Predictor(Xtest), Ytest))
     
    # return the cross validation estimate of predicion error by averaging the 
    # predictions errors of each fold
    return 1/k * sum(predictionErrors)

def missclassifactionError(Yhat, Ytrue):
    N = {}
    
    # N will hold the number of elements in eah class. Used to wieight the 
    # 0-1 loss function. 
    for i in set(Ytrue):
        N[i] = Ytrue.count(i)
        
    loss = lambda yhat, ytrue: 1 / N[ytrue] if yhat != ytrue else 0
    return sum([loss(Yhat[i], Ytrue[i]) for i in range(len(Ytrue))])

"""
function: logRegression(Xtrain, Ytrain, preprocess, Xtest)

parameters:  Xtrain: Observations to train on
             Ytrain: Responses to train on
             preprocess: Method for preprocessing data. returns a preprocessor 
                         function to transform data
             Xtest: 

Returns: The predicted output of the model.

Note: This method is curried, i.e. it is partially callable. 
"""
@pymonad.curry
def logRegression(Xtrain, YTrain, preprocess, Xtest):
    preprocessor = preprocess(Xtrain)
    
    Xp = preprocessor(Xtrain)
    
    logReg = LogisticRegression()
    logReg.fit(Xp,YTrain)
    
    return logReg.predict(preprocessor(Xtest))

"""
function: KNNPCAPreProcess(Xtrain, X)

parameters: Xtrain: Observations to standardize and train on
            X: Observations to be transformed using the same scaling and 
            feature space reduction methods as Xtrain
            
Returns: Transformed X

Note: This method is curreid, i.e. it is partially callable
"""
@pymonad.curry
def KNNPCAPreProcess(XTrain, X):
    XTrainFilled = KNNImpute(k=5).complete(XTrain)
    XFilled = KNNImpute(k=5).complete(X)

    scaler = StandardScaler().fit(XTrainFilled)
    
    XScaledFilled = scaler.transform(XFilled).tolist()
    XTrainScaledFilled = scaler.transform(XTrainFilled).tolist()
    
    dimReducer = PCA(n_components=int(math.floor(len(XTrain[0]) / 4)))
    dimReducer.fit(XTrainScaledFilled)
    
    return dimReducer.transform(XScaledFilled)
    
    
"""
function: KNNHierClusterPreProcess(Xtrain, X)

parameters: Xtrain: Observations to standardize and train on
            X: Observations to be transformed using the same scaling and 
            feature space reduction methods as Xtrain
            
Returns: Transformed X

Note: This method is curreid, i.e. it is partially callable
"""
@pymonad.curry
def KNNHierClusterPreProcess(XTrain, X):
    XTrainFilled = KNNImpute(k=5).complete(XTrain)
    XFilled = KNNImpute(k=5).complete(X)

    scaler = StandardScaler().fit(XTrainFilled)
    
    XScaledFilled = scaler.transform(XFilled).tolist()
    XTrainScaledFilled = scaler.transform(XTrainFilled).tolist()
    
    dimReducer = FeatureAgglomeration(n_clusters=int(math.floor(len(XTrain[0]) / 3)))
    dimReducer.fit(XTrainScaledFilled)
    
    return dimReducer.transform(XScaledFilled)
    
    

def ee445final():
    data = []
    X = []
    Y = []
    
    with open('/Users/charlesdickens/Documents/GitHub/EE445Final/AMLtrain.csv', 'r') as csvfile:
       reader = csv.reader(csvfile, delimiter='\n', quotechar='|')
       regex = re.compile(',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*\Z)')
       for row in reader:
          data.append(re.split(regex,row[0]))

    # feature we are predicitng
    Y = [d[13] for d in data[1:]]
    # not interested in patient id and we remove the features we are predicting
    X = [d[1:13] + d[18:] for d in data[1:]]
    
    # binarize categorical/discrete features
    X = binarizeCategoricalData(X, data)
    
    """ 
    logistic regression:
        KNN imputation
        hierarchical clustering for feature agglomeration.
        10-fold cross validation.
    """
    
   
    """ 
    logistic regression:
        KNN imputation
        Principal Component Analysis for feature space dimension reduction.
        10-fold cross validation.
    """
     
    
    """ 
    Random Forests:
        KNN imputation
        hierarchical clustering for feature agglomeration.
        10-fold cross validation.
    """
   
    """ 
    Random Forests:
        KNN imputation
        Principal Component Analysis for feature space dimension reduction.
        10-fold cross validation.
    """
   
    
    """ 
    SVM:
        KNN imputation
        hierarchical clustering for feature agglomeration.
        10-fold cross validation.
    """
   
    """ 
    SVM:
        KNN imputation
        Principal Component Analysis for feature space dimension reduction.
        10-fold cross validation.
    """
   
    """
    model selection
    """
    