#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:24:00 2018

@author: charlesdickens
"""

from __future__ import division
from fancyimpute import KNN as KNNImpute
import csv    
from sklearn.preprocessing import MultiLabelBinarizer

def ee445final():
    data = []
    X = []
    Y = []
    
    with open('/Users/charlesdickens/Documents/GitHub/EE445Final/AMLtrain.csv', 'r') as csvfile:
       reader = csv.reader(csvfile, delimiter='\n', quotechar='|')
       for row in reader:
          data.append(row[0].split('"'))

    # feature we are predicitng
    Y = [d[13] for d in data[1:]]
    # not interested in patient id and we remove the features we are predicting
    X = [d[1:13] + d[18:] for d in data[1:]]
    
    # binarize categorical/discrete features
    mlb = mlb = MultiLabelBinarizer()
    
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
    X = [X[i][0:25] + list(mlb.fit_transform([[d[9]] for d in data[1:]])[i]) + X[i][27:] for i in range(len(X))]
    