# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:38:15 2019

@author: falcon1
"""
from GO import getGOSet
from Bio import SeqIO
import json
import numpy as np
import pickle as pk

def expressProteinByGO():
    i = 1
    prot_dict = {}
    for seq_record in SeqIO.parse('e:\\Repoes\\SubLoc.fasta', 'fasta'):
        goes = getGOSet(str(seq_record.seq))
        prot_dict[seq_record.id] = list(goes)
        print(i)
        i = i + 1
    
    with open('e:\\Repoes\\SubLocGOExp.json','a') as fw:
       json.dump(prot_dict, fw,ensure_ascii=False)
       fw.write('\n')
       

def markProteinSubcellLoc():
    mark = ['Q96M11','Q6IBS0','O95866','Q96BM9','P47710','O95866','P11279',\
            'Q9NYL5','Q16762','Q6ZYL4','Q86WA8','O95866','P60880']
    pidList = []
    pseqList = []
    for record in SeqIO.parse('SubLoc.fasta','fasta'):
        pidList.append(record.id)
        pseqList.append(str(record.seq))
    
    classStartIndex=[0]    
    for m in mark:
        classStartIndex.append(pidList.index(m))
    
    classStartIndex[6] = pidList.index(mark[5],974)
    classStartIndex[8] = pidList.index(mark[7],1108)
    classStartIndex[12] = pidList.index(mark[11],1612)
    classStartIndex.append(3681)
    
    classesName = ['entrosome','cytoplasm','cytoskeleton','endoplasmic reticulum',\
                   'endosome','extracellular','Golgi apparatus','lysosome',\
                   'microsome','mitochondrion','nucleus','peroxisome','plasma membrane',\
                   'synapse']
    
    fp = open('SubLocGOExp.json','r')
    prot_dict = json.load(fp)
    fp.close()
    
    prot_data={}
    prot_heads=[]
    prot_seqs=[]
    prot_labels=np.zeros(shape=[3106,14])
    prot_goexpress=[]
    j = 0
    
    for i in range(len(pidList)):
        
        k = 0
        while(i>classStartIndex[k]):
            k = k + 1
        # 如果蛋白质不在数据集中，添加蛋白质到数据集
        # 分别记录下蛋白质访问的id,序列，go表示集合，标签    
        if pidList[i] not in prot_heads:
            prot_heads.append(pidList[i])#protein's acc id
            prot_seqs.append(pseqList[i])#protein's sequence
            
            goes = prot_dict[pidList[i]]
            goList = []
            for g in goes:
                goList.append(g[0])
            prot_goexpress.append(goList)#protein's gene ontology set
            
            prot_labels[j][k-1] = 1#protein's label
            j = j+1
        else: # 如果蛋白质已经在数据集中，首先找到其位置，然后相应地标记标签
            ind = prot_heads.index(pidList[i])
            prot_labels[ind][k-1] = 1
            
    # 保存数据
    prot_data['heads'] = prot_heads
    prot_data['seqs'] = prot_seqs
    prot_data['goes'] = prot_goexpress
    prot_data['labels'] = prot_labels.tolist()
    
    with open('subcellLocData.json','w') as fw:
        json.dump(prot_data,fw)
#if __name__ == '__main__':
#    main()