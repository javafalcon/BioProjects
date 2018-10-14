# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 09:52:10 2018

@author: falcon1
"""

import tflearn
from tflearn.data_utils import to_categorical
from generatorBenchmark import loadDataFromCSV
from sklearn.model_selection import train_test_split

# DNA binding sites Dataset loading
X, y = loadDataFromCSV('../data/datasets_ws15.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Converting labels to binary vectors
y_train = trainY = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)

X_train = X_train.reshape([-1,31,5])
X_test = X_test.reshape([-1,31,5])

net = tflearn.input_data(shape=[None,31,5], name='input')
net = tflearn.conv_1d(net, 32, 3, activation='relu', regularizer='L2')
net = tflearn.max_pool_1d(net,2)
#net = tflearn.local_response_normalization(net)
net = tflearn.conv_1d(net, 64, 3, activation='relu', regularizer='L2')
net = tflearn.max_pool_1d(net,2)
#net = tflearn.local_response_normalization(net)
net = tflearn.fully_connected(net,128, activation='tanh')
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, name='target')

model = tflearn.DNN(net)
model.fit({'input':X_train},{'target': y_train}, n_epoch=32,
           validation_set=({'input':X_test},{'target':y_test}),
           snapshot_step=100, show_metric=True, run_id='convnet_dbs')
