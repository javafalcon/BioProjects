# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:54:07 2018

@author: falcon1
"""
import numpy as np
import json

text='ARNDCQEGHILKMFPSTWYVX'
properties = ['Molecular Weight','Hydrophobicity','pK1','pK2','PI']
codeDict = {}
codeDict['Molecular Weight'] = [89.09, 174.20, 132.12, 133.10, 121.15, 146.15, 147.13, 75.07, 155.16, 131.17, \
			   131.17, 146.19, 149.21, 165.19, 115.13, 105.09, 119.12, 204.24, 181.19, 117.15,0]
codeDict['Hydrophobicity'] = [0.87, 0.85, 0.09, 0.66, 1.52, 0, 0.67, 0.1, 0.87, 3.15,\
		       2.17, 1.64, 1.67, 2.87, 2.77, 0.07, 0.07, 3.77, 2.67, 1.87,0]
codeDict['Hydrophobicity-JACS'] = [0.62, -2.53, -0.78, -0.90, 0.29, -0.85, -0.74, 0.48, -0.40, 1.38,\
				1.06, -1.50, 0.64, 1.19, 0.12, -0.18, -0.05, 0.81, 0.26, 1.08,0]
codeDict['pK1'] = [2.35, 2.18, 2.18, 1.88, 1.71, 2.17, 2.19, 2.34, 1.78, 2.32, \
				2.36, 2.20, 2.28, 2.58, 1.99, 2.21, 2.15, 2.38, 2.20, 2.29,0]
codeDict['pK2'] = [9.87, 9.09, 9.09, 9.60, 10.78, 9.13, 9.67, 9.60, 8.97, 9.76, \
				9.60, 8.90, 9.21, 9.24, 10.60, 9.15, 9.12, 9.39, 9.11, 9.74,0]
codeDict['PI'] = [6.11, 10.76, 10.76, 2.98, 5.02, 5.65, 3.08, 6.06, 7.64, 6.04, \
				6.04, 9.47, 5.74, 5.91, 6.30, 5.68, 5.60, 5.88, 5.63, 6.02,0]
def stdCode(): 
    stdCode = {}
    for s in properties:
        m = np.array(codeDict[s])
        mean = m.mean()
        dev = m.std()
        stdCode[s] = (m - mean)/dev
    return stdCode


def scalCode():
    scalCodes = {}
    for s in properties:
        m = np.array(codeDict[s])
        zmax,zmin = m.max(),m.min()
        scalCodes[s] = (m - zmin)/(zmax - zmin)
    return scalCodes

def sigmCode():
    sigmCodes = {}
    stdCodes = stdCode()
    for s in properties:
        m = np.array(stdCodes[s])
        sigmCodes[s] = 1/(1+np.exp(-m))
    return sigmCodes

    
def saveCode2Json( c, file):
    #stdCode = stdcode()
    aalist = list(text)
    AACode = {}
    for s in properties:
        AACode[s] = dict(zip(aalist, c[s]))
        #print(AAStdCode[s])
    #fw = open("../data/AAStdCode.json","w")
    fw = open(file, "w")
    json.dump(AACode,fw,indent=4)
    fw.close()   

def loadCode(file):
    #fr = open("../data/AAStdCode.json","r")
    fr = open(file,"r")
    AACode = json.load(fr)
    fr.close()
    return AACode

def main():
    sigmCodes = sigmCode()
    saveCode2Json(sigmCodes, "../data/AASigmCode.json")
    aac = loadCode("../data/AASigmCode.json")
    print(aac)
    
if __name__ == "__main__":
    main()
        