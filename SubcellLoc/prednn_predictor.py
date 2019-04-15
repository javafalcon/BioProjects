#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:23:20 2019

@author: weizhong
"""
import json
import numpy as np
import re
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
    X=X.reshape((3106,224,224,3))   
    return X,y