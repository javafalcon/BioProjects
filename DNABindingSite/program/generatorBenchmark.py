# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:19:52 2018

@author: falcon1
"""
import json
from AAcoder import loadStdCode
import numpy as np

ROW = 31
AAStdCode = loadStdCode()
keys = ['Molecular Weight','Hydrophobicity','pK1','pK2','PI']
def load():
    fr = open('../data/seqdata.json','r')
    seqs = json.load(fr)
    fr.close()
    positive = seqs['positive']
    negative = seqs['negative']
    return positive,negative
def code(seq):
    mat = np.ndarray((ROW,5))
    i,j = 0, 0
    for c in seq:
        j = 0
        for key in keys:
            mat[i][j] = AAStdCode[key][c]
            j += 1
        i += 1
    return mat
def formulateSeqs():
    positiveSeq,negativeSeq = load()
    len_of_positive = len(positiveSeq)
    len_of_negative = len(negativeSeq)
    positiveMatrix = np.ndarray((len_of_positive,ROW,5))
    negativeMatrix = np.ndarray((len_of_negative,ROW,5))
    i = 0
    for seq in positiveSeq:
        positiveMatrix[i] = code(seq)
        i += 1
    i = 0
    for seq in negativeSeq:
        negativeMatrix[i] = code(seq)
        i += 1
    return (positiveMatrix, negativeMatrix)
    
     
