# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:17:37 2019

@author: falcon1
"""

import json
from GO import goTermsDist
import numpy as np

#distance of proteins
def distance(goset1:set, goset2:set):
    """
    计算两个蛋白质之间的距离，蛋白质由它的GO特征方法表示
    Parameters:
    -----------
    goset1: set
    goset2: set
    """
    if not goset1 or not goset2:
        return 0
        
    a = goset1.union(goset2)
    b = goset1.intersection(goset2) 
    
    s1 = goset1.difference(b)
    s2 = goset2.difference(b)
    
    if not s1 or not s2:
        return len(b)/len(a)
    
    k, dist = 0, 0
    for g1 in s1:
        for g2 in s2:
           dist = dist + goTermsDist(g1,g2) 
           k = k + 1
    dist = dist/k + len(b)/len(a)
    return dist

def distMetrix(X:list):
    """
    计算机数据集中两两样本之间的距离
    Parameters
    ----------
    X: list
    
    Return:
    ---------
    numpy.ndarray
        距离矩阵
    """
    f = open("distance.txt",'a')
    N = len(X)
    d = np.ndarray(shape=[N,N])
    for i in range(N):
        d[i][i] = -1
        for j in range(i+1,N):
            print("calculte distance(%s,%s)"%(i,j))
            d[i][j] = distance(set(X[i]),set(X[j]))
            d[j][i] = d[i][j]
            f.write(str(d[i][j]))
            f.write(",")
        f.write("\n")
    return d
            
def compute_prior(s,y):
    """Helper function to compute for the prior probabilities

    Parameters
    ----------
    y : numpy.ndarray
        the training labels

    Returns
    -------
    numpy.ndarray
        the prior probability given true
    numpy.ndarray
        the prior probability given false
    """
    prior_prob_true = np.array((s + y.sum(axis=0)) / (s * 2 + y.shape[0]))
    prior_prob_false = 1 - prior_prob_true

    return (prior_prob_true, prior_prob_false)    

def compute_cond(X, y, s, k):
    """Helper function to compute for the posterior probabilities

    Parameters
    ----------
    X : go terms set
    y : numpy.ndaarray
        binary indicator matrix with label assignments.
    k:  int
        the number of nearest neighbors
    Returns
    -------
    numpy.ndarray
        the posterior probability given true
    numpy.ndarray
        the posterior probability given false
    """
    num_instance, num_class = y.shape
    
    # 每个样本的邻居按距离从小到大排序。排在第一个近邻是其自身。
    NeighborsIndex = np.ndarray(shape=[num_instance, num_instance],dtype=np.int)    
    dist_matrix = distMetrix(X)
    for i in range(num_instance):
        NeighborsIndex[i] = np.argsort(dist_matrix[i])
        
    c = np.zeros(shape=[num_class, k],dtype=int)
    cn = np.zeros(shape=[num_class, k], dtype=int)
    
    # 统计每个样本的K近邻中属于标签j-th的样本数
    for i in range(num_instance):
        neighbor_labels = y[NeighborsIndex[1:,k+1]]
        temp = np.sum(neighbor_labels)
        for j in num_class:
            if y[i][j] == 1:
                c[j][temp[j]] = c[j][temp[j]] + 1
            else:
                cn[j][temp[j]] = cn[j][temp[j]] + 1
                
    cond_prob_true = np.ndarray(shape=[num_class,k])  
    cond_prob_false = np.ndarray(shape=[num_class,k])          
    
    # 计算后验概率
    for i in range(num_class):
        temp1 = sum(c[i])
        temp2 = sum(cn[i])
        for j in range(k):
            cond_prob_true[i][j] = (s + c[i][j])/(s*(k+1) + temp1)
            cond_prob_false[i][j] = (s + cn[i][j])/(s*(k+1) + temp2)
    
    return cond_prob_true, cond_prob_false

def compute_posterior(X,y,s,k):
    # num_instance: 样本个数
    # num_class: 标签数
    num_instance, num_class = y.shape
    
    # cond_prob_true[i][j]: 后验概率
    #    事件“样本标记为i类，且它的k近邻中有j(j=0,1,2,...,k)个样本标记为i类”发生的概率
    cond_prob_true = np.ndarray[num_class, k+1]
    
    # cond_prob_false[i][j]: 后验概率
    #    事件“样本没有标记为i类，但它的k近邻中有j(j=0,1,2,...,k)个样本标记为i类”发生的概率
    cond_prob_false = np.ndarray[num_class, k+1]
    
    # 每个样本的邻居按距离从小到大排序。排在第一个近邻是其自身。
    NeighborsIndex = np.ndarray(shape=[num_instance, num_instance],dtype=np.int)    
    dist_matrix = distMetrix(X)
    for i in range(num_instance):
        NeighborsIndex[i] = np.argsort(dist_matrix[i])
        

    for m in range(num_class):
        # 标记为m类样本的k近邻中有0,1,2,...,k个样本属于第m类这一事件的发生次数
        c = np.zeros([k+1,1],dtype=int)
        # 没有标记为m类样本的k近邻中有0,1,2,...,k个样本属于第m类这一事件的发生次数
        cn = np.zeros([k+1,1],dtype=int) 
        
        for i in range(num_instance):
            # 统计第i个样本的K近邻中标记为第m类的样本数
            delta = 0
            for j in range(1,k+1):
                if y[NeighborsIndex[i][j]][m] == 1:
                    delta = delta + 1
                    
            # 已知第i个样本标记为第m类，它有delta个近邻样本也标记为m类，这一事件发生的次数
            if y[i][m] == 1:
                c[delta] = c[delta] + 1
            # 已知第i个样本没有标记为第m类，它有delta个近邻样本标记为m类，这一事件发生的次数
            else:
                cn[delta] = cn[delta] + 1
         
        
        for j in range(k+1):
            cond_prob_true[m][j] = (s + c[j]) / (s*(k+1) + np.sum(c))
            cond_prob_false[m][j] = (s + cn[j]) / (s*(k+1) + np.sum(cn))
            
    return cond_prob_true, cond_prob_false    

def train(X,y,s,k):
    """
    Fit classifier with training data
    Parameters
    ----------
    X : list
        input features
    y : numpy.ndarray
        binary indicator matrix with label assignments
    s : smooth coeficient
    k : the number of nearest neighbor
    """
    prior_prob_true, prior_prob_false = compute_prior(s,y)
    cond_prob_true, cond_prob_false = compute_posterior(X,y,s,k)
    return (prior_prob_true, prior_prob_false, cond_prob_true, cond_prob_false)

def predict(X_test, X_train, y_train, k, prior_prob_true, prior_prob_false, cond_prob_true, cond_prob_false):
    num_train, num_class = y_train.shape
    num_test = len(X_test)
    predict_label = np.zeros([num_test, num_class])
    predict_prob = np.ndarray([num_test, num_class])
    
    for t in range(num_test):
        neighbors = np.ndarray([num_train,1])
        for i in range(num_train):
            neighbors[i] = distance(set(X_test[t]), set(X_train[i]))
        neighborsIndex = np.argsort(neighbors)
        
        for m in range(num_class):
            # 统计待预测的样本t在训练集的k近邻中，标记了m类的样本数
            c = 0
            for n in range(k):
                if y_train[neighborsIndex[n]][m] == 1:
                    c = c + 1
            prob_true = prior_prob_true[m]*cond_prob_true[m][c]
            prob_false = prior_prob_false[m]*cond_prob_false[m][c]
            
            if prob_true > prob_false:
                predict_label[t][m] = 1
            
            predict_prob[t][m] = prob_true/(prob_true + prob_false)
            
    return (predict_label, predict_prob)
    
# load data
with open('subcellLocData.json','r') as fr:
    prot=json.load(fr)
X = prot['goes']
dist = distMetrix(X)
