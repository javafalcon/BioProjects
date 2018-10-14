# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:15:44 2018

@author: falcon1
"""

import tflearn
from tflearn.data_utils import to_categorical
from generatorBenchmark import loadDataFromCSV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
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
net = tflearn.input_data([None, 31,5])
#net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X_train, y_train, validation_set=(X_test, y_test), show_metric=True,
          batch_size=32)

# print confusion matrix
y_pred_list = model.predict(X_test)
ind = [ np.argmax(a) for a in y_pred_list]
y_pred = np.zeros( ( len(y_pred_list), 2))
y_pred[:,ind] = 1   
print(confusion_matrix(y_test[:,1],y_pred[:,1]))