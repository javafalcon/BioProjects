# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:12:13 2019

@author: falcon1
"""
import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten,concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    with open('subcellLocData.json','r') as fr:
         prot=json.load(fr)
    seqs = prot['seqs']
    token = Tokenizer(char_level=True)
    token.fit_on_texts(seqs)
    x=token.texts_to_sequences(seqs)
    x=pad_sequences(x,maxlen=5000,padding='post',truncating='post')
    y = np.array(prot['labels'])
    return x,y

def net():
    l2value = 0.001
    main_input = Input(shape=(5000,), dtype='int32', name='main_input')
    x = Embedding(output_dim=200, input_dim=23, input_length=5000)(main_input)
    a = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_size=5, padding="same", strides=1)(a)
    b = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_size=5, padding="same", strides=1)(b)
    c = Conv1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_size=5, padding="same", strides=1)(c)
    d = Conv1D(64, 9, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    dpool = MaxPooling1D(pool_size=5, padding="same", strides=1)(d)
    f = Conv1D(64, 4, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    fpool = MaxPooling1D(pool_size=5, padding="same", strides=1)(f)
    g = Conv1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    gpool = MaxPooling1D(pool_size=5, padding="same", strides=1)(g)
    h = Conv1D(64, 6, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    hpool = MaxPooling1D(pool_size=5, padding="same", strides=1)(h)
    i = Conv1D(64, 7, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    ipool = MaxPooling1D(pool_size=5, padding="same", strides=1)(i)
    merge2 = concatenate([apool, bpool, cpool, dpool,fpool,gpool,hpool, ipool], axis=-1)
    merge2 = Dropout(0.3)(merge2)
    scalecnn1 = Conv1D(64, 11, activation='relu', padding='same', kernel_regularizer=l2(l2value))(merge2)
    scale1 = MaxPooling1D(pool_size=5, padding="same", strides=1)(scalecnn1)
    scalecnn2 = Conv1D(64, 13, activation='relu', padding='same', kernel_regularizer=l2(l2value))(merge2)
    scale2 = MaxPooling1D(pool_size=5, padding="same", strides=1)(scalecnn2)
    scalecnn3 = Conv1D(64, 15, activation='relu', padding='same', kernel_regularizer=l2(l2value))(merge2)
    scale3 = MaxPooling1D(pool_size=5, padding="same", strides=1)(scalecnn3)
    scale = concatenate([scale1, scale2, scale3], axis=-1)
    scale = Dropout(0.3)(scale)
    cnn1 = Conv1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(scale)
    cnn10 = MaxPooling1D(pool_size=5, padding="same", strides=1)(cnn1)
    cnn2 = Conv1D(64, 9, activation='relu', padding='same', kernel_regularizer=l2(l2value))(scale)
    cnn20 = MaxPooling1D(pool_size=5, padding="same", strides=1)(cnn2)
    cnn3 = Conv1D(64, 13, activation='relu', padding='same', kernel_regularizer=l2(l2value))(scale)
    cnn30 = MaxPooling1D(pool_size=5, padding="same", strides=1)(cnn3)
    cnn50 = concatenate([cnn10, cnn20, cnn30], axis=-1)
    cnn50 = Dropout(0.3)(cnn50)
    x = Flatten()(cnn50)
    x = Dense(256, activation='relu', name='FC', kernel_regularizer=l2(l2value))(x)
    output = Dense(14,activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)
    model = Model(main_input, output)
    
    return model

X,y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
lr = 0.001 # Learning rate
adam = Adam(lr=lr)
model = net()
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
#best_Weight_File="/name_of_the_weight_File.hdf5"
#checkpoint = ModelCheckpoint(best_Weight_File, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callback_list = [checkpoint]
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64)
y_pred=model.predict(X_test)    
    
y_p = np.array(y_pred > 0.5).astype(int)
print("subAcc=%f"%(accuracy_score(y_test,y_p)))