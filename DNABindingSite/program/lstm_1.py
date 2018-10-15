# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:17:46 2018

@author: falcon1
"""

import tflearn
from tflearn.data_utils import to_categorical
from generatorBenchmark import loadDataFromCSV
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN


# long short Term Memory Recurrent networks
def lstm(trainX, testX, trainY, testY):
    # Converting labels to binary vectors
    testY = to_categorical(testY,2)
    trainY = to_categorical(trainY,2)
    # data preprocessing
    trainX = trainX.reshape([-1,31,5])
    testX = testX.reshape([-1,31,5])
    # Network building
    net = tflearn.input_data([None, 31,5])
    #net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
    # Predicting
    model.load('model.tflearn')
    y_pred_list = model.predict(testX)
    
    return y_pred_list


# print confusion matrix
def printConfusionMatrix(y_pred_list, y_test):
    y_pred = np.zeros( ( len(y_pred_list), 2))
    for i in range(len(y_pred_list)):
        if y_pred_list[i][0] > y_pred_list[i][1]:
            y_pred[i][0] = 1
        else:
            y_pred[i][1] = 1
    print(confusion_matrix(y_test[:,1],y_pred[:,1]))    
    
    
# DNA binding sites Dataset loading
X, y = loadDataFromCSV('../data/datasets_std_ws15.csv')

# stratified shuffle split data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
for train_indx, test_indx in sss.split(X,y):
    X_train, X_test = X[train_indx], X[test_indx]
    y_train, y_test = y[train_indx], y[test_indx]

# oversampling
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)

# Converting labels to binary vectors
testY = to_categorical(y_test,2)
trainY = to_categorical(y_train_resampled,2)
# data preprocessing
testX = X_test.reshape([-1,31,5])
trainX = X_train_resampled.reshape([-1,31,5])

#network building
net = tflearn.input_data([None, 31,5])
#net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.7, activation='relu', return_seq=True)
net = tflearn.lstm(net, 128)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
# Training 
model = tflearn.DNN(net, tensorboard_verbose=2)

model.fit(trainX, trainY, n_epoch=20, 
          validation_set=(testX, testY), 
          show_metric=True,
          batch_size=50, shuffle=True)
    
pred_list = model.predict(testX)
    
printConfusionMatrix(pred_list, testY)
