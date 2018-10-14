# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:15:44 2018

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

# data preprocessing
X_train = X_train.reshape([-1,31,5])
X_test = X_test.reshape([-1,31,5])

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