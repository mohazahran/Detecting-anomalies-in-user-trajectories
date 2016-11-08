#-*- coding: utf8
from __future__ import division, print_function

from statsmodels.distributions.empirical_distribution import ECDF

import matplotlib
#matplotlib.use('Agg')

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('model_fpath', help='The name of the model file (a h5 file)', \
    #        type=str)
    #args = parser.parse_args()
    model_fpath = '/home/zahran/Desktop/shareFolder/PARSED_pins_repins_win10_noop_NoLeaveOut_pinterest.h5'
    model = pd.HDFStore(model_fpath, 'r')    
    outPath = model_fpath + '_ENV_USER_INFO'
    w = open(outPath, 'w') 
         
    
    assign = model['assign'].values[:, 0] #a vector of the env id used for each line in the trainingData. ie. its length=len(to-from)
    Theta_zh = model['Theta_zh'].values
    Theta_hz = Theta_zh.T * model['count_z'].values[:, 0]
    Theta_hz = Theta_hz / Theta_hz.sum(axis=0)

    Psi_oz = model['Psi_sz'].values
    hyper2id = model['hyper2id'].values
    source2id = model['source2id'].values
    
    from collections import Counter
    counter = Counter(assign) # dict. keys are envId and values, is the freq of these envId
    
    id2hyper = dict((r[1], r[0]) for r in hyper2id)
    id2source = dict((r[1], r[0]) for r in source2id)
    
    nz = Psi_oz.shape[1] 
    k = 10
    for z, pz in counter.most_common()[-nz:]:
        print(z)
        w.write('\nEnv: '+str(z))
        
        print('These Users\n--')
        w.write('\nTop users:')
        for i in Theta_hz[:, z].argsort()[::-1][:k]:
            print(id2hyper[i], Theta_hz[i, z])
            w.write('\n\tUser: '+str(id2hyper[i])+' prob: '+str(Theta_hz[i, z]))
        print()
        
        w.write('\nTop items:')
        print('Transition Through These Objects\n--')
        for i in Psi_oz[:, z].argsort()[::-1][:k]:
            print(id2source[i], Psi_oz[i, z])
            w.write('\n\tItem: '+str(id2source[i])+' prob: '+str(Psi_oz[i, z]))
        print()
        print()
        w.write('\n')

    model.close()
    w.close()

if __name__ == '__main__':
    main()
