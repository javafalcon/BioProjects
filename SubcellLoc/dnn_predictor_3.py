#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:03:47 2019

@author: weizhong
"""
from time import time
import json
import re
import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNEL = 3

def AA_phychem_code():
    """
    aa = "ARNDCQEGHILKMFPSTWYV"
    """
    norm = {}
    ind = {}
    """
    COWR900101 hydrophobicity
    Hydrophobicity indices for amino acid residues as determined by hight-
    performance liquid chromatography, Peptide Res. 3, 75-80(1990)
    """
    ind["hydrophobicity"]=  [0.42,-1.56,-1.03,-0.51,0.84,-0.96,-0.37,0,-2.28,1.81,1.80,-2.03,
           1.18,1.74,0.86,-0.64,-0.26,1.46,0.51,1.34]
    
    
    """
    PONJ960101 volumes
    eviations from standard atomic volumes as a quality measure for protein 
    crystal structures, J. Mol. Biol 264, 121-136 (1996) 
    """
    ind["volumes"] = [91.5, 196.1, 138.3, 135.2, 114.4, 156.4, 154.6, 67.5, 163.2, 162.6,
           163.4, 162.5, 165.9, 198.8, 123.4, 102.0, 126.0, 209.8, 237.2, 138.4]
    
    """
    RADA880106 surface
    """
    ind["surface"] = [93.7, 250.4, 146.3, 142.6, 135.2, 177.7, 182.9, 52.6, 188.1, 182.2,
           173.7, 215.2, 197.6, 228.6, 0, 109.5, 142.1, 271.6, 239.9, 157.2]
    
    for key in ind.keys():
        indx = np.array(ind[key])
        mean = indx.mean()
        deviation = indx.std()
        st = (indx - mean) / deviation
        norm[key] = st
    return norm
    
def load_data():
    amino_acid = 'ARNDCQEGHILKMFPSTWYV'
    norm = AA_phychem_code()
    with open('subcellLocData.json','r') as fr:
        prot=json.load(fr)
    seqs = prot['seqs']
        
    X = np.ndarray((3106,50176,3))
    y = np.array(prot['labels'])
    
    for k in range(3106):
        i = 0
        seq = seqs[k]
        seq = re.sub('[XZUB]',"",seq)
        seq = seq.strip()
        for ch in seq:
            indx = amino_acid.index(ch)
            X[k][i][0] = norm["hydrophobicity"][indx]
            X[k][i][1] = norm["volumes"][indx]
            X[k][i][2] = norm["surface"][indx]
            i = i + 1
            if i == 50176:
                break
    X=X.reshape((-1,224,224,3))   
    return X,y

def co_label_mat(y):
    c = np.zeros((14,14))
    for yy in y:
        for i in range(14):
            for j in range(14):
                if yy[i] == 1 and yy[j] == 1:
                    c[i][j] = c[i][j] + 1
    s = sum(c)
    for i in range(14):
        if s[i] != 0:
            c[i] = c[i]/s[i]                
    c = c.reshape((1,196))
    return c
    
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='bias')

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def avg_pool(x, k):
    return tf.nn.avg_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

# 
def buildNet(X_train, y_train, X_test, y_test, epochs=25, batch_size=64):
    X_train, y_train = shuffle(X_train, y_train)
    #input layer
    with tf.name_scope('input_layer'):
        x = tf.placeholder(tf.float32, 
                           shape=[None,INPUT_WIDTH,INPUT_HEIGHT,INPUT_CHANNEL],
                           name='X')
        x_co = tf.placeholder(tf.float32, 
                           shape=[None,14,14,1],
                           name='X')
        keep_prob = tf.placeholder(tf.float32)
    # The first connv layer
    with tf.name_scope('conv_1'):
        w1 = weight([3,3,INPUT_CHANNEL,32])
        b1 = bias([32])
        conv_1 = conv2d(x, w1) + b1
        conv_1 = tf.nn.relu(conv_1)
        conv_1 = tf.nn.l2_normalize(conv_1,dim=1)
    # The first pool layer
    with tf.name_scope('pool_1'):
        pool_1 = avg_pool(conv_1, 2)
    # The first droupout layer
    with tf.name_scope('drop_1'):
        drop_1 = tf.nn.dropout(pool_1, keep_prob)
        
    # The 2-th connv layer
    with tf.name_scope('conv_2'):
        w2 = weight([3,3,32,64])
        b2 = bias([64])
        conv_2 = conv2d(drop_1, w2) + b2
        conv_2 = tf.nn.relu(conv_2)
        conv_2 = tf.nn.l2_normalize(conv_2, dim=1)
    # The 3-th connv layer
    with tf.name_scope('conv_3'):
        w3 = weight([3,3,64,64])
        b3 = bias([64])
        conv_3 = conv2d(conv_2, w3) + b3
        conv_3 = tf.nn.relu(conv_3)
        conv_3 = tf.nn.l2_normalize(conv_3, dim=1)
    # The 2-th pool layer
    with tf.name_scope('pool_2'):
        pool_2 = avg_pool(conv_3, 2)  
    # The 2-th droupout layer
    with tf.name_scope('drop_2'):
        drop_2 = tf.nn.dropout(pool_2, keep_prob)
        
    # The 4-th connv layer
    with tf.name_scope('conv_4'):
        w4 = weight([3,3,64,128])
        b4 = bias([128])
        conv_4 = conv2d(drop_2, w4) + b4
        conv_4 = tf.nn.relu(conv_4)
        conv_4 = tf.nn.l2_normalize(conv_4, dim=1)
    # The 5-th connv layer
    with tf.name_scope('conv_5'):
        w5 = weight([3,3,128,128])
        b5 = bias([128])
        conv_5 = conv2d(conv_4, w5) + b5
        conv_5 = tf.nn.relu(conv_5)
        conv_5 = tf.nn.l2_normalize(conv_5, dim=1)
    # The 3-th pool layer
    with tf.name_scope('pool_3'):
        pool_3 = avg_pool(conv_5, 2)  
    # The 3-th droupout layer
    with tf.name_scope('drop_3'):
        drop_3 = tf.nn.dropout(pool_3, keep_prob)
        
    # The first set of FC
    with tf.name_scope('fc_1'):
        w6 = weight([28*28*128, 1024])
        b6 = bias([1024])
        flat_1 = tf.reshape(drop_3, [-1,28*28*128])
        h1 = tf.nn.relu(tf.matmul(flat_1,w6) + b6)
        h1 = tf.nn.l2_normalize(h1, dim=1)
        h1 = tf.nn.dropout(h1, 0.5)
    # The 2-th set of FC
    with tf.name_scope('fc2'):
        w7 = weight([1024, 2048])
        b7 = bias([2048])
        h2 = tf.nn.relu(tf.matmul(h1, w7) + b7)
        h2 = tf.nn.l2_normalize(h2, dim=1)
        h2 = tf.nn.dropout(h2, keep_prob)
    
        
    # The first colabel connv layer
    with tf.name_scope('colabel_conv_1'):
        w1_co = weight([3,3,1,32])
        b1_co = bias([32])
        conv_1_co = conv2d(x_co, w1_co) + b1_co
        conv_1_co = tf.nn.relu(conv_1_co)
        conv_1_co = tf.nn.l2_normalize(conv_1_co, dim=1)
    # The 2-th colabel connv layer
    with tf.name_scope('colabel_conv_2'):
        w2_co = weight([3,3,32,64])
        b2_co = bias([64])
        conv_2_co = conv2d(conv_1_co, w2_co) + b2_co
        conv_2_co = tf.nn.relu(conv_2_co)
        conv_2_co = tf.nn.l2_normalize(conv_2_co, dim=1)
        
    # The first colabel set of FC
    with tf.name_scope('colabel_fc_1'):
        w3_co = weight([14*14*64, 256])
        b3_co = bias([256])
        flat_1_co = tf.reshape(conv_2_co, [-1,14*14*64])
        h1_co = tf.nn.relu(tf.matmul(flat_1_co,w3_co) + b3_co)
        h1_co = tf.nn.l2_normalize(h1_co, dim=1)
        h1_co = tf.nn.dropout(h1_co, keep_prob)
    # The 2-th colabel set of FC
    with tf.name_scope('colabel_fc_2'):
        w4_co = weight([256, 500])
        b4_co = bias([500])
        h2_co = tf.nn.relu(tf.matmul(h1_co, w4_co) + b4_co)
        h2_co = tf.nn.l2_normalize(h2_co, dim=1)
        h2_co = tf.nn.dropout(h2_co, keep_prob)    
        
        
    net = tf.concat([h2,h2_co],axis=1)
    
    with tf.name_scope("concat_fc1"):
        w1_concat = weight([2048+500, 500])
        b1_concat = bias([500])
        fc_concat = tf.nn.relu(tf.matmul(net, w1_concat) + b1_concat)
    # output layer
    with tf.name_scope('output_layer'):
        w_out = weight([500,14])
        b_out = bias([14])
        pred = tf.nn.sigmoid(tf.matmul(fc_concat, w_out) + b_out)
    
    # net model
    with tf.name_scope("optimizer"):
        y = tf.placeholder(tf.float32, shape=[None, 14], name='label')
        loss_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(loss_function)
        
    '''with tf.name_scope("evalulation"):
        threshold = tf.constant(0.5, shape=[None, 14], name="threshold")
        correct_prediction = tf.greater(pred, threshold)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))'''
        
    X_TRAIN_SIZE = len(X_train)
    TRAIN_EPOCHES = epochs
    BATCH_SIZE = batch_size
    TOTAL_BATCH = int( np.ceil( X_TRAIN_SIZE / BATCH_SIZE))
    epoch= tf.Variable(0, name='epoch', trainable=False)
    STARTTIME = time()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 设置检查点存储目录
        ckpt_dir = "../log/"
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        #生成saver
        saver = tf.train.Saver(max_to_keep=5)
        # 创建 summary_writer，用于写图文件
        summary_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)
        # 如果有检查点文件，读取最新的检查点文件，恢复各种变量值
        ckpt = tf.train.latest_checkpoint(ckpt_dir )
        if ckpt != None:
            saver.restore(sess, ckpt)     
        else:
            print("Training from scratch.")

        start_epoch= sess.run(epoch)
        print("Training starts form {} epoch.".format(start_epoch+1))
        
        #迭代训练
        for ep in range(start_epoch, start_epoch + TRAIN_EPOCHES):
            for i in range(TOTAL_BATCH):
                start = (i * BATCH_SIZE) % X_TRAIN_SIZE
                end = min(start + BATCH_SIZE, X_TRAIN_SIZE)
                batch_x = X_train[start:end]
                batch_y = y_train[start:end]
                co_y = co_label_mat(batch_y)
                co_y = np.tile(co_y,len(batch_x))
                co_y = co_y.reshape((-1,14,14,1))
                sess.run(optimizer,feed_dict={x: batch_x, x_co: co_y, y: batch_y, keep_prob:0.75})
                if i % 100 == 0:
                    print("Step {}".format(i), "finished")

            loss = sess.run([loss_function],feed_dict={x: batch_x, x_co: co_y, y: batch_y, keep_prob:0.75})
            
            print("Train epoch:", '%02d' % (sess.run(epoch)+1), \
                  "Loss=","{:.6f}".format(loss))

            #保存检查点
            saver.save(sess,ckpt_dir+"DBPSite_cnn_model.cpkt",global_step=ep+1)

            sess.run(epoch.assign(ep+1))
    
        duration =time()-STARTTIME
        print("Train finished takes:",duration)   
    
        #计算测试集上的预测结果
        y_co = co_label_mat(y_train)
        y_co = np.tile(y_co, len(y_test))
        y_co = y_co.reshape([-1,14,14,1])
        y_pred = sess.run(pred, feed_dict={x: X_test, x_co: y_co, y: y_test, keep_prob:1.0})
        
    return y_pred

X,y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
buildNet(X_train,y_train,X_test,y_test)