import re
import os
import numpy as np
from scipy.sparse import coo_matrix
from Bio import SeqIO
import tensorflow as tf
from time import time
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

#把一个长度为L的氨基酸序列转换为一个矩阵型数组。按照20个氨基酸的两联体以ONE-HOT方式编码
#矩阵的列数是441
#矩阵的行数根据参数r，有：(L-1)+(L-2)+...+(L-r)=rL-(r+1)*r/2
def seq2DaaOneHotArray(sequence, r):
    L = len(sequence)
    N = r*L - ((r+1)*r)/2
    result = np.zeros(shape=(int(N), 441))
    m = 0
    for i in range(r):
        for j in range(L-i-1):
            aa = sequence[j]+sequence[j+i+1]
            k = daa.index(aa)
            result[m][k] = 1
            m = m + 1
    return result


# 读入序列文件和位点文件
def loadBindingsites(fastaFile, siteFile):
    # 读序列文件，每一个序列构成字典的一项，
    # key：序列的id
    # value: 氨基酸序列的字母字符串
    data = {}
    for seq_record in SeqIO.parse(fastaFile, 'fasta'):
        data[seq_record.id] = seq_record.seq

    # 读位点文件
    bindingsites = {}
    with open(siteFile, 'r') as pbsreader:
        i = 0
        for line in pbsreader:
            i = i + 1
            line = line.strip()
            if '>' in line:
                sid = line[1:]
            else:
                sites = line.split()
                bs = []
                for site in sites:
                    bs.append(int(site))
            if i % 2 == 0:
                bindingsites[sid] = bs

    return (data, bindingsites)


# 构建序列样本集
# 把binding site和not binding site分别以稀疏矩阵存放在正负数据集中
def splitDatasets(data, bindingsites, ws, r):
    positive = []
    negative = []
    for key in data:
        sites = bindingsites[key]
        p = data[key]
        seqlen = len(p)
        for j in range(seqlen):
            if j < ws:
                seq = str(p[j - ws:]) + str(p[0: ws + j + 1])
            elif j > seqlen - ws - 1:
                seq = str(p[j - ws:j]) + str(p[j:]) + str(p[0:ws - seqlen + j + 1])
            else:
                seq = str(p[j - ws:j + ws + 1])
            m = seq2DaaOneHotArray(seq, r)
            sm = coo_matrix(m)
            if j in sites:
                positive.append(sm)
            else:
                negative.append(sm)

     return (positive, negative)

def preprocess(seqfile, sitefile, ws=15, r=15):

    #data, bindingsites = loadBindingsites('../data/PDNA-224.fasta', '../data/PDNA-224-binding-sites.txt')
    data, bindingsites = loadBindingsites(seqfile, sitefile)
    positive, negative = splitDatasets(data, bindingsites, ws, r)

daa = []
for x in alphabet:
    for y in alphabet:
        daa.append(x + y)

seqfile = '../data/PDNA-224.fasta'
sitefile = '../data/PDNA-224-binding-sites.txt'
positive,negative = preprocess(seqfile,sitefile)
