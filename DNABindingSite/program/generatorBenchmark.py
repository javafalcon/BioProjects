# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:19:52 2018

@author: falcon1
"""
import json
from AAcoder import loadCode
import numpy as np
import csv

ROW = 31
AACode = loadCode("../data/AASigmCode.json")
keys = ['Molecular Weight','Hydrophobicity','pK1','pK2','PI']
def loadSeqs():
    fr = open('../data/seqdata.json','r')
    seqs = json.load(fr)
    fr.close()
    positive = seqs['positive']
    negative = seqs['negative']
    return positive,negative
def aaCode(seq):
    mat = np.ndarray((ROW,5))
    i,j = 0, 0
    for c in seq:
        j = 0
        for key in keys:
            mat[i][j] = AACode[key][c]
            j += 1
        i += 1
    return mat
def formulateSeqs():
    positiveSeq,negativeSeq = loadSeqs()
    len_of_positive = len(positiveSeq)
    len_of_negative = len(negativeSeq)
    positiveMatrix = np.ndarray((len_of_positive,ROW,5))
    negativeMatrix = np.ndarray((len_of_negative,ROW,5))
    i = 0
    for seq in positiveSeq:
        positiveMatrix[i] = aaCode(seq)
        i += 1
    i = 0
    for seq in negativeSeq:
        negativeMatrix[i] = aaCode(seq)
        i += 1
    #return (positiveMatrix, negativeMatrix)
    X = np.concatenate((positiveMatrix, negativeMatrix))
    print(X.shape)
    X = np.reshape(X, (57348,155))#31*5=155
    plen = positiveMatrix.shape[0]
    nlen = negativeMatrix.shape[0]
    Y1 = np.ones((plen,1))
    Y2 = np.zeros((nlen,1))
    Y = np.concatenate((Y1,Y2))

    print(X.shape)
    print(Y.shape)
    data = np.hstack((X, Y))
    with open('../data/datasets_sigm_ws15.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def loadDataFromCSV(csvfile):
    list_X = []
    list_Y = []
    with open(csvfile,'r', newline='') as cf:
        reader = csv.reader(cf)
        for row in reader:
            rd = []
            for e in row[0:-1]:
                rd.append(eval(e))
            list_X.append(rd)
            list_Y.append(eval(row[-1]))
    return (np.array(list_X), np.array(list_Y))
        
def main():
    formulateSeqs()
    '''X,Y = loadDataFromCSV('../data/datasets_ws15.csv')
    print(X[0])
    print(Y[0])'''
    
if __name__ == '__main__':
    main()
     
