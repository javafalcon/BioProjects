# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 14:46:01 2018

@author: Administrator
"""

import numpy as np
import csv
'''a=np.zeros((3,4))
b=np.ones((5,4))
c=np.concatenate((a,b))
with open('d:/example.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(c)'''
with open('d:/example.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)