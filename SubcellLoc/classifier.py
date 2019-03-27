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
    a = goset1.union(goset2)
    b = goset1.intersection(goset2)
    dist = 0
    s1 = goset1.difference(b)
    s2 = goset2.difference(b)
    k = 0
    for g1 in s1:
        for g2 in s2:
           dist = dist + goTermsDist(g1,g2) 
           k = k + 1
    dist = dist/k + len(b)/len(a)
    return dist

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

def compute_cond(X, y):
        """Helper function to compute for the posterior probabilities

        Parameters
        ----------
        X : go terms set
        y : numpy.ndaarray
            binary indicator matrix with label assignments.

        Returns
        -------
        numpy.ndarray
            the posterior probability given true
        numpy.ndarray
            the posterior probability given false
        """

        self.knn_ = NearestNeighbors(self.k).fit(X)
        c = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='i8')
        cn = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='i8')

        label_info = get_matrix_in_format(y, 'dok')

        neighbors = [a[self.ignore_first_neighbours:] for a in
                     self.knn_.kneighbors(X, self.k + self.ignore_first_neighbours, return_distance=False)]

        for instance in range(self._num_instances):
            deltas = label_info[neighbors[instance], :].sum(axis=0)
            for label in range(self._num_labels):
                if label_info[instance, label] == 1:
                    c[label, deltas[0, label]] += 1
                else:
                    cn[label, deltas[0, label]] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='float')
        cond_prob_false = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='float')
        for label in range(self._num_labels):
            for neighbor in range(self.k + 1):
                cond_prob_true[label, neighbor] = (self.s + c[label, neighbor]) / (
                        self.s * (self.k + 1) + c_sum[label, 0])
                cond_prob_false[label, neighbor] = (self.s + cn[label, neighbor]) / (
                        self.s * (self.k + 1) + cn_sum[label, 0])
        return cond_prob_true, cond_prob_false
    
# load data
with open('subcellLocData.json','r') as fr:
    prot=json.load(fr)

