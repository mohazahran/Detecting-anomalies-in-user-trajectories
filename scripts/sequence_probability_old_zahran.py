'''
Created on Jul 21, 2016

@author: zahran
'''
#-*- coding: utf8
from __future__ import division, print_function

from tribeflow import _eval
from tribeflow.mycollections.stamp_lists import StampLists

import tribeflow
import pandas as pd
import plac
import numpy as np
import math

def predictObject(store, sequenceLength, tsFile, traceFile, true_mem_size, hyper2id, obj2id, previous_stamps, Theta_zh, Psi_sz, count_z, kernel):    
    
    
    tsLine = tsFile.readline() 
    l = traceFile.readline()       
    spl = l.strip().split('\t')
    parts = tsLine.strip().split('\t')    
    seqProb = 0.0
    
    for targetObjIdx in range(1,sequenceLength): #targetObjIdx=0 cannot be predicted we have to skip it
        HSDs = []
        Dts = []               
        mem_size = targetObjIdx         
        dts_line = [float(x) for x in spl[:mem_size-1]]
        currentTime = parts[targetObjIdx]
        dts_line.append(currentTime)
        h = spl[true_mem_size]
        #d = spl[-1]
        d = spl[true_mem_size + targetObjIdx + 1]
        #true_sources = spl[true_mem_size + 1:-1]
        sources = spl[true_mem_size + 1: true_mem_size + 1 + targetObjIdx + 1]
        
        all_in = h in hyper2id and d in obj2id
        for s in sources:
            all_in = all_in and s in obj2id
        
        if all_in:
            trace_line = [hyper2id[h]] + [obj2id[s] for s in sources] + [obj2id[d]]
            HSDs.append(trace_line)                
            Dts.append(dts_line)  
            HSDs = np.array(HSDs, dtype='i4')[[0]].copy()
            Dts = np.array(Dts, dtype='d')[[0]].copy()             
            rrs, preds = _eval.reciprocal_rank(Dts, HSDs, previous_stamps, Theta_zh, Psi_sz, count_z, kernel, True)
            targetObjId = obj2id[d]
            predTargetProb = preds[0,targetObjId]
            seqProb += math.log(predTargetProb)
            
    
#     num_queries = min(10000, len(HSDs))
#     queries = np.random.choice(len(HSDs), size=num_queries)
#     
    
  
    
#     np.savetxt(out_fpath_rrs, rrs)
#     np.savetxt(out_fpath_pred, preds)
#     print(rrs.mean(axis=0))
    store.close()

def main(model, out_fpath_rrs, out_fpath_pred):    
    store = pd.HDFStore(model)    
    trace_fpath = store['trace_fpath'][0][0]
    tsFile = open('/home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_B10_zahran_sampledData','r')
    sequenceLength = 10
    
    from_ = store['from_'][0][0]
    to = store['to'][0][0]
    assert from_ == 0
    
    kernel_class = store['kernel_class'][0][0]
    kernel_class = eval(kernel_class)

    Theta_zh = store['Theta_zh'].values
    Psi_sz = store['Psi_sz'].values
    count_z = store['count_z'].values[:, 0]
    P = store['P'].values
    residency_priors = store['residency_priors'].values[:, 0]    
    previous_stamps = StampLists(count_z.shape[0]) #previous_stamps has a length = nz

    true_mem_size = store['Dts'].values.shape[1]    
    tstamps = store['Dts'].values[:, 0] #tstamps (#trainingLines,) contains the  t(xi)-t(xi-1)
    assign = store['assign'].values[:, 0]# assign (#trainingLines,) each dim has the env id used in that training instance
    for z in xrange(count_z.shape[0]):
        idx = assign == z
        previous_stamps._extend(z, tstamps[idx]) #tstamps[idx]: is the tstamps whose corresponding index in idx is True

    hyper2id = dict(store['hyper2id'].values)
    obj2id = dict(store['source2id'].values)
    trace_size = sum(count_z) #sum of the number of appearances of all envs
    kernel = kernel_class()
    kernel.build(trace_size, count_z.shape[0], residency_priors)
    kernel.update_state(P)
            
    with open(trace_fpath) as traceFile:  
        predictObject(store, sequenceLength, tsFile, traceFile, true_mem_size, hyper2id, obj2id, previous_stamps, Theta_zh, Psi_sz, count_z, kernel)
    tsFile.close()
    traceFile.close()
    
    
    
plac.call(main)
print('DONE!')