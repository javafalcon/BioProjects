# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:54:26 2018

@author: falcon1
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from generatorBenchmark import loadDataFromCSV
from sklearn.model_selection import train_test_split

'''
# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

# Converting labels to binary vectors
trainY = to_categorical(trainY,2)
testY = to_categorical(testY,2)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
'''

# DNA binding sites Dataset loading
X, y = loadDataFromCSV('../data/datasets_ws15.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Converting labels to binary vectors
y_train = trainY = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)
# Network building
net = tflearn.input_data([None, 155])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X_train, y_train, validation_set=(X_test, y_test), show_metric=True, batch_size=32 )
