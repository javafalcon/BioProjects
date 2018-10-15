# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 09:52:10 2018

@author: falcon1
"""

import tflearn
from tflearn.data_utils import to_categorical
from generatorBenchmark import loadDataFromCSV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix

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
positiveData = []
negativeData = []
datasize = len(X)
for i in range(datasize):
    if y[i] == 1:
        positiveData.append(X[i,:])
    else:
        negativeData.append(X[i,:])

# Construct testing dataset
test_positive = positiveData[-778:]
test_negative = negativeData[-778:]
train_positive = positiveData[0:-778]
train_negative = negativeData[0:-778]
testX_p = np.array(test_positive)
testX_n = np.array(test_negative)

testX = np.concatenate((testX_p, testX_n))
testY = np.concatenate((y[:778], y[-778:]))
# Converting labels to binary vectors
testY = to_categorical(testY,2)
# data preprocessing
testX = testX.reshape([-1,31,5])

# construct training dataset
trainX_p = np.array( positiveData[:-778])
indx = [i*3000 for i in range(17)]

net = tflearn.input_data(shape=[None,31,5], name='input')
net = tflearn.conv_1d(net, 128, 3, activation='relu', regularizer='L2')
net = tflearn.max_pool_1d(net,2)
#net = tflearn.local_response_normalization(net)
net = tflearn.conv_1d(net, 128, 3, activation='relu', regularizer='L2')
net = tflearn.max_pool_1d(net,2)

net = tflearn.conv_1d(net, 128, 3, activation='relu', regularizer='L2')
net = tflearn.max_pool_1d(net,2)
#net = tflearn.local_response_normalization(net)
net = tflearn.fully_connected(net,128, activation='tanh')
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, name='target')

model = tflearn.DNN(net)
for i in range(17):
    if i == 16:
        trainX_n = np.array( negativeData[indx[i]:])
        nsize = len(trainX_n)
    else:
        trainX_n = np.array(negativeData[indx[i]:indx[i+1]])
        nsize = 3000
        
    trainX = np.concatenate((trainX_p, trainX_n))
    trainY = np.concatenate( ( y[:3000], y[-nsize:]) ) 
    
    # Converting labels to binary vectors
    trainY = to_categorical(trainY,2)
    # data preprocessing
    trainX = trainX.reshape([-1,31,5])
    
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
    
pred_list = model.predict(testX)
    
printConfusionMatrix(pred_list, testY)