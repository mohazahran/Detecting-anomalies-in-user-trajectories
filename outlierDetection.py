#-*- coding: utf8
'''
Created on Aug 9, 2016

@author: zahran
'''

from __future__ import division, print_function

from tribeflow import _eval_zahran
from tribeflow.mycollections.stamp_lists import StampLists

import tribeflow
import pandas as pd
import plac
import numpy as np
import math
from collections import OrderedDict

ALPHA = 0.05


#def evaluate(HOs, Theta_zh, Psi_sz, count_z, env, candidate):  
def evaluate(userId, history, targetObjId, Theta_zh, Psi_sz, env):        
    mem_factor = 1.0    
    candidateProb = 0.0                    
    for j in xrange(len(history)):#for all B
        #i.e. multiply all psi[objid1,z]*psi[objid2,z]*..psi[objidB,z]
        mem_factor *= Psi_sz[history[j], env] # Psi[objId, env z]            
    mem_factor *= 1.0 / (1 - Psi_sz[history[len(history)-1], env])# 1-Psi_sz[mem[B-1],z] == 1-psi_sz[objIdB,z]       
    candidateProb += mem_factor * Psi_sz[targetObjId, env] * Theta_zh[env, userId]                                              
    return candidateProb
    


def calculateSequenceProb(h, theSequence, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z):                     
    seqProb = 0.0
    window = min(true_mem_size, len(theSequence))
    for z in xrange(Psi_sz.shape[1]): #for envs
        seqProbZ = 1.0        
        for targetObjIdx in range(0,len(theSequence)): #targetObjIdx=0 cannot be predicted we have to skip it
            if(targetObjIdx == 0):                
                d = theSequence[targetObjIdx]
                all_in = h in hyper2id and d in obj2id
                if(all_in):
                    targetObjId = obj2id[d]
                    prior = Psi_sz[targetObjId, z]
                    seqProbZ *= prior
            else:                                                                            
                d = theSequence[targetObjIdx]                 
                sources = theSequence[max(0,targetObjIdx-window): targetObjIdx] # look back 'window' actions.                             
                all_in = h in hyper2id and d in obj2id
                for s in sources:
                    all_in = all_in and s in obj2id
                
                if all_in:
                    targetObjId = obj2id[d]
                    userId = hyper2id[h]
                    history = [obj2id[s] for s in sources]                                
                    candProb = evaluate(userId, history, targetObjId, Theta_zh, Psi_sz, z) #(int[:, ::1] HOs, double[:, ::1] Theta_zh, double[:, ::1] Psi_sz, int[::1] count_z, int env):                                
                    seqProbZ *= candProb
                #seqProb += math.log(predTargetProb)
        seqProb += seqProbZ
        #print(seqProb, math.log(seqProb))
    return seqProb   

def bonferroni_hypothesis_testing(keySortedPvalues, pValues):
    outlierVector = ['N/A']*len(keySortedPvalues)
    bonferroni_ALPHA = ALPHA/len(keySortedPvalues)
    for i in range(len(keySortedPvalues)):
        if(pValues[keySortedPvalues[i]] <= bonferroni_ALPHA):
            outlierVector[keySortedPvalues[i]] = 'OUTLIER' # rejecting H0 (i.e rejecting that the action is normal ==> outlier)
        else:
            outlierVector[keySortedPvalues[i]] = 'NORMAL'
    return outlierVector

# def holm_hypothesis_testing (pValue, nTests, currentTestNumber):
#     holm_ALPHA = ALPHA/(nTests-currentTestNumber)
#     if(pValue < holm_ALPHA):
#         return 'OUTLIER'
#     return 'NORMAL'

def holm_hypothesis_testing (keySortedPvalues, pValues):
    k = -1
    outlierVector = ['N/A']*len(keySortedPvalues)  
    for i in range(len(keySortedPvalues)):
        val = ALPHA/(len(keySortedPvalues)-i)
        if(pValues[keySortedPvalues[i]] > val):
            k = i
            break
    for i in range(len(keySortedPvalues)):
        if(i<k):
            outlierVector[keySortedPvalues[i]] = 'OUTLIER'
        else:
            outlierVector[keySortedPvalues[i]] = 'NORMAL'
    return outlierVector
            
            
            
def getPvalueWithoutRanking(currentActionRank, keySortedProbs, probabilities):
    #normConst = 0.0
    #for i in range(len(probabilities)):
    #    normConst += probabilities[i]
        
    cdf = 0.0
    for i in range(currentActionRank+1):
        cdf += probabilities[keySortedProbs[i]]
    
    #prob = cdf/normConst
    return cdf
        
        
        
        
    
        
                             

def outlierDetection(SEQ_FILE_PATH, store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z):
    mylog = open(SEQ_FILE_PATH+'_ANAOMLY_ANALYSIS','w')
    seqFile = open(SEQ_FILE_PATH, 'r')
    for tsLine in seqFile: #for all test samples      
        #tsLine = '1707158\t21\t50\t5\t20\t19'        
        tmp = tsLine.strip().split('\t') 
        user = tmp[0]
        #mylog.write('\n'+tsLine)
        if(user not in hyper2id):            
            mylog.write('User not found in trainingSet !\n')
            #print(tsLine, ' User not found!')
            continue
        seq = tmp[1:] 
        actions = obj2id.keys()                  
        pValuesWithRanks = {}
        pValuesWithoutRanks = {}
        for i in range(len(seq)): #for all actions in the sequence.
            #Take the action with index i and replace it with all possible actions             
            probabilities = {}
            scores = {}        
            newSeq = list(seq)                        
            currentActionId = obj2id[newSeq[i]] #current action id
            currentActionIndex = actions.index(newSeq[i])# the current action index in the action list.
            #cal scores (an un-normalized sequence prob in tribeflow)
            normalizingConst = 0
            for j in range(len(actions)): #for all possible actions that can replace the current action
                del newSeq[i]                
                newSeq.insert(i, actions[j])                
                seqScore = calculateSequenceProb(user, newSeq, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z)
                scores[j] = seqScore
                normalizingConst += seqScore
            #cal probabilities
            for j in range(len(actions)): #for all possible actions that can replace the current action
                probabilities[j] = scores[j]/normalizingConst
            #sorting ascendingly
            keySortedProbs = sorted(probabilities, key=lambda k: (-probabilities[k], k), reverse=True)
            currentActionRank = keySortedProbs.index(currentActionIndex)
            currentActionPvalueWithoutRanks = getPvalueWithoutRanking(currentActionRank, keySortedProbs, probabilities)
            currentActionPvalueWithRanks = float(currentActionRank+1)/float(len(actions))
            pValuesWithRanks[i] = currentActionPvalueWithRanks
            pValuesWithoutRanks[i] = currentActionPvalueWithoutRanks
                    
        keySortedPvaluesWithRanks = sorted(pValuesWithRanks, key=lambda k: (-pValuesWithRanks[k], k), reverse=True)
        keySortedPvaluesWithoutRanks = sorted(pValuesWithoutRanks, key=lambda k: (-pValuesWithoutRanks[k], k), reverse=True)

        outlierVector_bonferroniWithRanks = bonferroni_hypothesis_testing(keySortedPvaluesWithRanks, pValuesWithRanks)
        outlierVector_bonferroniWithoutRanks = bonferroni_hypothesis_testing(keySortedPvaluesWithoutRanks, pValuesWithoutRanks)
        #outlierFlag = holm_hypothesis_testing(pValues[k], len(seq), idx)
        outlierVector_holmsWithRanks = holm_hypothesis_testing(keySortedPvaluesWithRanks, pValuesWithRanks)
        outlierVector_holmsWithoutRanks = holm_hypothesis_testing(keySortedPvaluesWithoutRanks, pValuesWithoutRanks)
        
        print(tmp)
        mylog.write('userId: '+str(tmp[0])+'\n')
        mylog.write('Action pvalue_with_ranks bonferroni_withRanks holms_withRanks pValues_withoutRanks holms_withRanks bonferroni_withoutRanks holms_withoutRanks\n')
        for x in range(0,len(tmp)-1):
            mylog.write('||'+str(tmp[x+1])+'|| ')
            mylog.write(str(pValuesWithRanks[x])+' ')
            mylog.write(str(outlierVector_bonferroniWithRanks[x])+' ')
            mylog.write(str(outlierVector_holmsWithRanks[x])+' ')
            mylog.write(str(pValuesWithoutRanks[x])+' ')
            mylog.write(str(outlierVector_bonferroniWithoutRanks[x])+' ')
            mylog.write(str(outlierVector_holmsWithoutRanks[x])+' ')
            mylog.write('\n')
        mylog.write('\n')
        
        
            
            
        
        #mylog.write('pValues_withRanks: '+str(pValuesWithRanks)+'\n bonferroni_withRanks: '+str(outlierVector_bonferroniWithRanks)+'\n holms_withRanks: '+str(outlierVector_holmsWithRanks)+'\n')
        #mylog.write('pValues_withoutRanks: '+str(pValuesWithoutRanks)+'\n bonferroni_withoutRanks: '+str(outlierVector_bonferroniWithoutRanks)+'\n holms_withoutRanks: '+str(outlierVector_holmsWithoutRanks)+'\n')
        #print(tsLine, outlierVector, pValues)
        mylog.flush()
    mylog.close()
    seqFile.close()
                                            
def main():    
    store = pd.HDFStore('/home/zahran/Desktop/tribeFlow/zahranData/pinterest/PARSED_pinterest_model.h5')  
    #trace_fpath = store['trace_fpath'][0][0]
    SEQ_FILE_PATH = createTestingSeqFile(store)
    #SEQ_FILE_PATH = '/home/zahran/Desktop/tribeFlow/zahranData/pinterest/test_traceFile_win5'
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
    
    outlierDetection(SEQ_FILE_PATH, store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z)
            
    #calculateSequenceProb(SEQ_FILE_PATH, store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z)
    
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