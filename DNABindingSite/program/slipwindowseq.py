# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:16:24 2018

@author: Lin, Weizhong
"""

from Bio import SeqIO
import json

# 读入序列文件和位点文件
# 返回值：
#  -- data：字典类型，key=id, value=sequence
#  -- bindingsites: 字典类型，key=id，value是一个列表，列表中的元素是binding位点的索引（从1开始计数）
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
# 把binding site和not binding site分别保存在positive和negative样本集
# ws: 滑窗大小
def splitDatasets(data, bindingsites, ws):
    seqdata = {}
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
            
            if (j+1) in sites:
                positive.append(seq)
            else:
                negative.append(seq)

    seqdata['positive'] = positive
    seqdata['negative'] = negative
    print("lenght of positive:",len(positive))
    print("length of negative:",len(negative))
    return seqdata

def main():
    seqfile = '../data/PDNA-224.fasta'
    sitefile = '../data/PDNA-224-binding-sites.txt'
    data,bindingsites = loadBindingsites(seqfile, sitefile)
    seqdata = splitDatasets(data,bindingsites,15)
    jssd = json.dumps(seqdata)
    fileObject = open('../data/seqdata.json','w')
    fileObject.write(jssd)
    fileObject.close()
    
if __name__ == "__main__":
    main()
    