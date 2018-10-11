# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:35:22 2018

@author: falcon1
"""

import numpy as np
from scipy.sparse import coo_matrix
from Bio import SeqIO
import tensorflow as tf
from time import time
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#定义权值
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

#定义偏置
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='bias')

#定义卷积操作
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#定义最大池化操作
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#定义平均池化操作
def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#
def cnn(x_train_pos, x_train_neg, x_test_pos, x_test_neg):
    x_train, y_train = shuffle(x_train, y_train)
    #输入层 31-by-5
    with tf.name_scope('input_layer'):
        x = tf.placeholder(tf.float32,shape=[None, 31, 5, 1], name='x')
    #第1个卷积层
    with tf.name_scope('conv_1'):
        W1 = weight([3,3,1,32])
        b1 = bias([32])
        conv_1 = conv2d(x, W1) + b1
        conv_2 = tf.nn.relu(conv_1)

    #第1个池化层 16-by-11
    with tf.name_scope('pool_1'):
        pool_1 = avg_pool_2x2(conv_1)

    #第2个卷积层
    with tf.name_scope('conv_2'):
        W2 = weight([3,3,32,64])
        b2 = bias([64])
        conv_2 = conv2d(pool_1, W2) + b2
        conv_2 = tf.nn.relu(conv_2)

    #第2个池化层 8-by-6
    with tf.name_scope("pool_2"):
        pool_2 = avg_pool_2x2(conv_2)

    #第3个卷积层
    with tf.name_scope('conv_3'):
        W3 = weight([3,3,64,128])
        b3 = bias([128])
        conv_3 = conv2d(pool_2, W3) + b3
        conv_3 = tf.nn.relu(conv_3)

    #第3个池化层 4-by-3
    with tf.name_scope('pool_3'):
        pool_3 = avg_pool_2x2(conv_3)

    #全连接层
    with tf.name_scope('fc'):
        #将最后一个池化层的128个通道的4-by-3的图像转换为一维向量，长度是128*4*3=1536
        W4 = weight([1536,256]) #全连接层定义256个神经元
        b4 = bias([256])
        flat = tf.reshape(pool_3, [-1, 1536])
        h = tf.nn.relu(tf.matmul(flat, W4)) + b4
        keep_prob = tf.placeholder(tf.float32)
        h_dropout = tf.nn.dropout(h, keep_prob)

    #输出层
    with tf.name_scope('output_layer'):
        W5 = weight([256,2])
        b5 = bias([2])
        pred = tf.nn.softmax(tf.matmul(h_dropout, W5) + b5)
    
    #构建网络模型
    with tf.name_scope("optimizer"):
        #定义占位符
        y = tf.placeholder(tf.int32, shape=[None, 2], name="label")
        #定义损失函数
        loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,
                                                                                 labels=y))
        #选择优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss_function)
    
    #定义准确率
    with tf.name_scope("evalulation"):
        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    #训练模型
    X_TRAIN_SIZE = len(x_train)
    TRAIN_EPOCHES = 25
    BATCH_SIZE = 50
    TOTAL_BATCH = int( np.ceil( X_TRAIN_SIZE / BATCH_SIZE))
    epoch= tf.Variable(0, name='epoch', trainable=False)
    STARTTIME = time()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        #迭代训练
        for ep in range(start_epoch, start_epoch + TRAIN_EPOCHES):
            for i in range(TOTAL_BATCH):
                start = (i * BATCH_SIZE) % X_TRAIN_SIZE
                end = min(start + BATCH_SIZE, X_TRAIN_SIZE)
                batch_x = x_train[start:end]
                batch_y = y_train[start:end]
                sess.run(optimizer,feed_dict={x: batch_x, y: batch_y, keep_prob:0.7})
                if i % 100 == 0:
                    print("Step {}".format(i), "finished")

            loss,acc = sess.run([loss_function,accuracy],feed_dict={x: batch_x, y: batch_y, keep_prob:0.7})
            
            print("Train epoch:", '%02d' % (sess.run(epoch)+1), \
                  "Loss=","{:.6f}".format(loss)," Accuracy=",acc)

            #保存检查点
            saver.save(sess,ckpt_dir+"DBPSite_cnn_model.cpkt",global_step=ep+1)

            sess.run(epoch.assign(ep+1))
    
        duration =time()-STARTTIME
        print("Train finished takes:",duration)   
    
        #计算测试集上的预测结果
        y_pred = sess.run(pred, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
        
    return y_pred