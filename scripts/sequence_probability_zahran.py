'''
Created on Jul 21, 2016

@author: zahran
'''
#-*- coding: utf8
from __future__ import division, print_function

from tribeflow import _eval_zahran
from tribeflow.mycollections.stamp_lists import StampLists

import tribeflow
import pandas as pd
import plac
import numpy as np
import math


def evaluate(HOs, Theta_zh, Psi_sz, count_z, env, candidate):      
    ns = Psi_sz.shape[0]
    mem = np.zeros((HOs.shape[1]-1), dtype='i4') #mem.shape = (B,)    
    #mem_factor = np.zeros(Psi_sz.shape[1], dtype='d') #mem_factor.shape = (nz,)
    mem_factor = 1.0
    p = np.zeros(Psi_sz.shape[0], dtype='d')# p.shape = (no,)
    candidateProb = 0.0
                
    for i in xrange(HOs.shape[0]): # for all test instances        
        h = HOs[i, 0]
        for j in xrange(mem.shape[0]): # for B (excluding the last object which is tha to be predicted one)
            mem[j] = HOs[i, 1 + j] #mem will hold a list of the B history obj ids. i.e. mem[0] = id(ob1)       
       
        for j in xrange(mem.shape[0]):#for all B
            #i.e. multiply all psi[objid1,z]*psi[objid2,z]*..psi[objidB,z]
            mem_factor *= Psi_sz[mem[j], env] # Psi[objId, env z]            
            mem_factor *= 1.0 / (1 - Psi_sz[mem[mem.shape[0] - 1], env])# 1-Psi_sz[mem[B-1],z] == 1-psi_sz[objIdB,z]
       
        
        candidateProb += mem_factor * Psi_sz[candidate, env] * Theta_zh[env, h]
            #print (mem_factor, Psi_sz[candidate_o, env], Theta_zh[env, h])                                   
          
        return candidateProb
    


def calculateSequenceProb(SEQ_FILE_PATH, store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z):
    seqFile = open(SEQ_FILE_PATH, 'r')
    for tsLine in seqFile:        
        parts = tsLine.strip().split('\t')  
        h = parts[0]  
        seqProb = 0.0
        window = min(true_mem_size, len(parts)-1)
        for z in xrange(Psi_sz.shape[1]): #for envs
            seqProbZ = 1.0
            for targetObjIdx in range(2,len(parts)): #targetObjIdx=0 cannot be predicted we have to skip it
                HSDs = []                                                                
                d = parts[targetObjIdx] 
                #print(d)           
                sources = parts[max(1,targetObjIdx-window): targetObjIdx]
                #print(sources)
                
                all_in = h in hyper2id and d in obj2id
                for s in sources:
                    all_in = all_in and s in obj2id
                
                if all_in:
                    trace_line = [hyper2id[h]] + [obj2id[s] for s in sources] + [obj2id[d]]
                    HSDs.append(trace_line)                               
                    HSDs = np.array(HSDs, dtype='i4')[[0]].copy()                                        
                    targetObjId = obj2id[d]
                    #preds = _eval_zahran.reciprocal_rank(HSDs, Theta_zh, Psi_sz, count_z, z) #(int[:, ::1] HOs, double[:, ::1] Theta_zh, double[:, ::1] Psi_sz, int[::1] count_z, int env):                
                    candProb = evaluate(HSDs, Theta_zh, Psi_sz, count_z, z, targetObjId) #(int[:, ::1] HOs, double[:, ::1] Theta_zh, double[:, ::1] Psi_sz, int[::1] count_z, int env):                                
                    seqProbZ *= candProb
                    #seqProb += math.log(predTargetProb)
            seqProb += seqProbZ
        print(seqProb, math.log(seqProb))
    seqFile.close()
                       

def main(model, out_fpath_rrs, out_fpath_pred):    
    store = pd.HDFStore(model)    
    #trace_fpath = store['trace_fpath'][0][0]
    SEQ_FILE_PATH = createTestingSeqFile(store)
    #SEQ_FILE_PATH = '/home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/SEQ_try'
    #sequenceLength = 10
    
    from_ = store['from_'][0][0]
    to = store['to'][0][0]
    
    Theta_zh = store['Theta_zh'].values
    Psi_sz = store['Psi_sz'].values
    count_z = store['count_z'].values[:, 0]   
    true_mem_size = store['Dts'].values.shape[1]    
    hyper2id = dict(store['hyper2id'].values)
    obj2id = dict(store['source2id'].values)
    trace_size = sum(count_z) #sum of the number of appearances of all envs  
            
    calculateSequenceProb(SEQ_FILE_PATH, store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z)
    
    store.close()    
   

def createTestingSeqFile(store):
    from_ = store['from_'][0][0]
    to = store['to'][0][0]
    trace_fpath = store['trace_fpath']
    tpath = '/home/zahran/Desktop/tribeFlow/zahranData/pinterest/test_traceFile_win5'
    w = open(tpath,'w')
    r = open(trace_fpath[0][0],'r')
    
    cnt = 0
    for line in r:
        if(cnt > to):
            p = line.strip().split('\t')
            usr = p[4]
            sq = p[5:]
            w.write(str(usr)+'\t')
            for s in sq:
                w.write(s+'\t')
            w.write('\n')
        cnt += 1
    w.close()
    r.close()
    return tpath
            
    
    
    
    

plac.call(main)
print('DONE!')