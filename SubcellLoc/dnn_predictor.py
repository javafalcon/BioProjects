# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:30:28 2019

@author: falcon1
"""

import json
import numpy as np
from SeqFormulate import seqAAOneHot
from SeqFormulate import seqDAA
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Concatenate
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def load_data():
    with open('subcellLocData.json','r') as fr:
        prot=json.load(fr)
    list_seq = prot['seqs']
        
    X = np.ndarray((3106,100,20,2))
    y = np.array(prot['labels'])
    
    for k in range(3106):
        seq = list_seq[k]
        X[k,:,:,0] = seqAAOneHot(seq,0,100)
        X[k,:,:,1] = seqAAOneHot(seq,0,-100)
        
        
    return X,y

def kNN(x, X_train, k=1):
    """
    在训练集X_train中查找x的最近K邻，返回这k个近邻在训练集中的下标
    """
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X_train)
    ind = neigh.kneighbors(x, return_distance=False)
    return ind

def concat_kNN_Model():
    chanDim = -1
    input1 = Input(shape=[100,20,2])
    out1 = Conv2D(32,(3,3), padding="same", activation='relu')(input1)
    out1 = BatchNormalization(axis=-1)(out1)
    out1 = AveragePooling2D(pool_size=(2, 2))(out1)
    out1 = Dropout(0.25)(out1)

    # (CONV => RELU) * 2 => POOL
    out1 = Conv2D(64, (3, 3), padding="same",activation='relu')(out1)
    out1 = BatchNormalization(axis=chanDim)(out1)
    out1 = Conv2D(64, (3, 3), padding="same",activation='relu')(out1)
    out1 = BatchNormalization(axis=chanDim)(out1)
    out1 = AveragePooling2D(pool_size=(2, 2))(out1)
    out1 = Dropout(0.25)(out1)

    # (CONV => RELU) * 2 => POOL
    out1 = Conv2D(128, (3, 3), padding="same",activation='relu')(out1)
    out1 = BatchNormalization(axis=chanDim)(out1)
    out1 = Conv2D(128, (3, 3), padding="same",activation='relu')(out1)
    out1 = BatchNormalization(axis=chanDim)(out1)
    out1 = AveragePooling2D(pool_size=(2, 2))(out1)
    out1 = Dropout(0.25)(out1)

    # first set of FC => RELU layers
    out1 = Flatten()(out1)
    out1 = Dense(1024, activation='relu')(out1)
    out1 = BatchNormalization()(out1)
    out1 = Dropout(0.5)(out1)
    
    # second set of FC
    out1 = Dense(2048, activation='relu')(out1)
    out1 = BatchNormalization()(out1)
    out1 = Dropout(0.5)(out1)
    
    model1 = Model(input1, out1)
    
    #chanDim = 0
    input2 = Input(shape=[1,14,14])
    out2 = Conv2D(32,(3,3), padding="same", activation='relu')(input2)
    out2 = BatchNormalization()(out2)
    
    out2 = Conv2D(96,(3,3), padding="same", activation='relu')(out2)
    out2 = BatchNormalization()(out2)
    
    #out2 = Conv2D(128,(3,3), padding="same", activation='relu')(input2)
    #out2 = BatchNormalization(axis=chanDim)(out2)
    
    out2 = Flatten()(out2)
    out2 = Dense(4096, activation='relu')(out2)
    out2 = BatchNormalization()(out2)
    out2 = Dropout(0.5)(out2)
    
   
    out2 = Dense(1000, activation='relu')(out2)
    out2 = BatchNormalization()(out2)
    out2 = Dropout(0.5)(out2)
    
    model2 = Model(input2,out2)
    
    conc = Concatenate(axis=-1)([model1.output,model2.output])
    out = Dense(500, activation="relu")(conc)
    out = Dropout(rate=0.5)(out)
    out = Dense(units=14, activation="sigmoid")(out)
    
    model = Model([model1.input, model2.input], out)
    
    return model

def label_dependence_knn(X_test, X_train, y_train,k=5):
    """使用聚类方法计算标签的依赖矩阵"""
    n_samples = X_test.shape[0]
    y = np.empty((n_samples,1,14,14))
    
    for i in range(n_samples):
        ind = kNN(X_test[i].reshape((-1,4000)), X_train, k+1)
        
        dep_mat = np.zeros([14,14])
        for ix in ind[0][1:]:
            temp = y_train[ix]
            for m in range(14):
                for n in range(14):
                    if temp[m] == 1 and temp[n] == 1:
                        dep_mat[m][n] = dep_mat[m][n] + 1
        
        y[i,0,:,:] = dep_mat
    return y

def label_dependence_cluster(X_test, X_train, y_train,n_clusters):
    """使用聚类方法计算标签的依赖矩阵"""
    n_samples = X_test.shape[0]
    y = np.empty((n_samples,1,14,14))
    dep_mat = np.zeros([n_clusters,14,14])
    kmeans_model = KMeans(n_clusters,random_state=1).fit(X_train)
    kmeans_label = kmeans_model.labels_
    
    for i in range(X_train.shape[0]):
        k = kmeans_label[i]# X_train[i]聚类到第k类
        temp = y_train[k]
        for m in range(14):
            for n in range(14):
                if temp[m] == 1 and temp[n] == 1:
                    dep_mat[k][m][n] = dep_mat[k][m][n] + 1
        
    test_label = kmeans_model.predict(X_test)
    for i in range(n_samples):
        y[i] = dep_mat[test_label[i]]
  
    return y


def test(X_train, X_test, y_train, y_test, K):
    
    label_train = label_dependence_knn(X_train.reshape((-1,4000)), X_train.reshape((-1,4000)), y_train, K)
    label_test = label_dependence_knn(X_test.reshape((-1,4000)), X_train.reshape((-1,4000)), y_train, K)
    
    INIT_LR = 1e-3
    EPOCHS = 75
    
    # initiallize the optimizer
    opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    nn = concat_kNN_Model()
    nn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # compile the model using binary cross-entropy rather than
    # categorical cross-entropy -- this may seem counterintuitive for
    # multi-label classification, but keep in mind that the goal here
    # is to treat each output label as an independent Bernoulli distribution
    
    # train the network
    print("[INFO] training network...")
    
    nn.fit([X_train, label_train], y_train,epochs=20,batch_size=64,verbose=0)
    score=nn.evaluate([X_test,label_test],y_test)
    print(score)
    y_pred=nn.predict([X_test, label_test])    
    
    y_p = np.array(y_pred > 0.5).astype(int)
    print("k=%d, subAcc=%f"%(K,accuracy_score(y_test,y_p)))


X,y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size=0.33, random_state=42)
for k in range(5,22,2):
    test(X_train, X_test, y_train, y_test, k)