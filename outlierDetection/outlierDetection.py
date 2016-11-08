#-*- coding: utf8
'''
Created on Aug 9, 2016

@author: zahran
'''

#from __future__ import division, print_function
from scipy.stats import chisquare
from collections import OrderedDict
from multiprocessing import Process, Queue

import tribeflow
import pandas as pd
import plac
import numpy as np
import math
import os.path
import cProfile
import _eval_outlier

CORES = 2
MODEL_PATH = '/home/zahran/Desktop/shareFolder/PARSED_pins_repins_win10_noop_NoLeaveOut_pinterest.h5'
#SEQ_FILE_PATH = '/home/zahran/Desktop/shareFolder/sqlData_likes_full_info_fixed_ONLY_TRUE_friendship' 
SEQ_FILE_PATH = '/home/zahran/Desktop/shareFolder/PARSED_pins_repins_win10_pinterest_INJECTED'
TESTSET_COUNT_ADJUST = True
UNBIAS_CATS_WITH_FREQ = False
smoothingParam = 1.0   #smootihng parameter for unbiasing item counts.
STAT_FILE = '/home/zahran/Desktop/shareFolder/Stats'

def evaluate(userId, history, targetObjId, Theta_zh, Psi_sz, env):        
    mem_factor = 1.0    
    candidateProb = 0.0                    
    for j in xrange(len(history)):#for all B
        #i.e. multiply all psi[objid1,z]*psi[objid2,z]*..psi[objidB,z]
        mem_factor *= Psi_sz[history[j], env] # Psi[objId, env z]            
    #mem_factor *= 1.0 / (1 - Psi_sz[history[len(history)-1], env])# 1-Psi_sz[mem[B-1],z] == 1-psi_sz[objIdB,z]       
    candidateProb += mem_factor * Psi_sz[targetObjId, env] * Theta_zh[env, userId]                                              
    return candidateProb
 
def calculateSequenceProb(theSequence, true_mem_size, userId, obj2id, Theta_zh, Psi_sz):                     
    seqProb = 0.0
    window = min(true_mem_size, len(theSequence))
    for z in xrange(Psi_sz.shape[1]): #for envs
        seqProbZ = 1.0        
        for targetObjIdx in range(0,len(theSequence)): #targetObjIdx=0 cannot be predicted we have to skip it
            if(targetObjIdx == 0):                
                d = theSequence[targetObjIdx]
                #all_in = h in hyper2id and d in obj2id
                #if(all_in):
                targetObjId = obj2id[d]
                prior = Psi_sz[targetObjId, z]
                seqProbZ *= prior
            else:                                                                            
                d = theSequence[targetObjIdx]                 
                sources = theSequence[max(0,targetObjIdx-window): targetObjIdx] # look back 'window' actions.                             
                #all_in = h in hyper2id and d in obj2id
                #for s in sources:
                #    all_in = all_in and s in obj2id                
                #if all_in:
                targetObjId = obj2id[d]
                #userId = hyper2id[h]
                history = [obj2id[s] for s in sources]                                
                candProb = evaluate(userId, history, targetObjId, Theta_zh, Psi_sz, z) #(int[:, ::1] HOs, double[:, ::1] Theta_zh, double[:, ::1] Psi_sz, int[::1] count_z, int env):
                #candProb = 0.001                    
                seqProbZ *= candProb
                #seqProb += math.log(predTargetProb)
        seqProb += seqProbZ
        #print(seqProb, math.log(seqProb))
    return seqProb   
                           
def getPvalueWithoutRanking(currentActionRank, keySortedProbs, probabilities):
    #normConst = 0.0
    #for i in range(len(probabilities)):
    #    normConst += probabilities[i]
        
    cdf = 0.0
    for i in range(currentActionRank+1):
        cdf += probabilities[keySortedProbs[i]]
    
    #prob = cdf/normConst
    return cdf
               
def outlierDetection(testDic, quota, coreId, q, store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, smoothedProbs):
    myCnt = 0    
    writer = open(SEQ_FILE_PATH+'_SCORES_ANAOMLY_ANALYSIS_'+str(coreId),'w')
    
    for user in testDic:
        for testLine in testDic[user]:
            myCnt += 1
            tmp = testLine.strip().split('\t')        
            seq = tmp[1:true_mem_size+2]
            goldMarkers = tmp[true_mem_size+2:]            
            
             
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
                    userId = hyper2id[user]     
                    #npNewSeq = np.array(newSeq, dtype=str) 
                    newSeqIds = [obj2id[s] for s in newSeq]   
                    newSeqIds_np = np.array(newSeqIds, dtype = 'i4').copy()
                    seqScore = _eval_outlier.calculateSequenceProb(newSeqIds_np, len(newSeqIds_np), true_mem_size, userId, Theta_zh, Psi_sz)                                            
                    #seqScore = calculateSequenceProb(newSeq, true_mem_size, userId, obj2id, Theta_zh, Psi_sz)
                    if(UNBIAS_CATS_WITH_FREQ):
                        unbiasingProb = 1.0
                        for ac in newSeq:
                            if(ac in smoothedProbs):
                                unbiasingProb *= smoothedProbs[ac]                                         
                        seqScore = float(seqScore)/float(unbiasingProb)
                        
                    scores[j] = seqScore
                    normalizingConst += seqScore
                #cal probabilities
                for j in range(len(actions)): #for all possible actions that can replace the current action
                    probabilities[j] = float(scores[j])/float(normalizingConst)
                #sorting ascendingly
                keySortedProbs = sorted(probabilities, key=lambda k: (-probabilities[k], k), reverse=True)
                currentActionRank = keySortedProbs.index(currentActionIndex)
                currentActionPvalueWithoutRanks = getPvalueWithoutRanking(currentActionRank, keySortedProbs, probabilities)
                currentActionPvalueWithRanks = float(currentActionRank+1)/float(len(actions))
                pValuesWithRanks[i] = currentActionPvalueWithRanks
                pValuesWithoutRanks[i] = currentActionPvalueWithoutRanks
                                
            writer.write('user##'+str(user)+'||seq##'+str(seq)+'||PvaluesWithRanks##'+str(pValuesWithRanks)+'||PvaluesWithoutRanks##'+str(pValuesWithoutRanks)+'||goldMarkers##'+str(goldMarkers)+'\n')        
            #if(myCnt%100 == 0):            
            writer.flush()
            print('>>> proc: '+ str(coreId)+' finished '+ str(myCnt)+'/'+str(quota)+' instances ...')                
    writer.close()    
    #ret = [chiSqs, chiSqs_expected]
    #q.put(ret)                                          
                                            
def calculatingItemsFreq(trace_fpath, true_mem_size):
    smoothedProbs = {}    
    if os.path.isfile(STAT_FILE):
        r = open(STAT_FILE, 'r')
        for line in r:
            parts = line.strip().split('\t')                
            smoothedProbs[parts[0]] = float(parts[1])                    
        return smoothedProbs
    
    freqs = {}            
    r = open(trace_fpath)
    counts = 0
    for line in r:
        cats = line.strip().split('\t')[true_mem_size+1:]
        for c in cats:
            if(c in freqs):
                freqs[c] += 1
            else:
                freqs[c] = 1                
            counts += 1
    for k in freqs:
        prob = float(freqs[k]+ smoothingParam) / float(counts + (len(freqs) * smoothingParam))
        smoothedProbs[k] = prob
    
    w = open(STAT_FILE, 'w')
    for key in smoothedProbs:
        w.write(key+'\t'+str(smoothedProbs[key])+'\n')
    w.close()
    return smoothedProbs
                  
def createTestingSeqFile(store):
    from_ = store['from_'][0][0]
    to = store['to'][0][0]
    trace_fpath = store['trace_fpath']
    Dts = store['Dts']
    winSize = Dts.shape[1]
    tpath = '/home/zahran/Desktop/tribeFlow/zahranData/pinterest/test_traceFile_win5'
    w = open(tpath,'w')
    r = open(trace_fpath[0][0],'r')
    
    cnt = 0
    for line in r:
        if(cnt > to):
            p = line.strip().split('\t')
            usr = p[winSize]
            sq = p[winSize+1:]
            w.write(str(usr)+'\t')
            for s in sq:
                w.write(s+'\t')
            w.write('\n')
        cnt += 1
    w.close()
    r.close()
    return tpath                                            
                                                                                        
def distributeOutlierDetection():
    store = pd.HDFStore(MODEL_PATH)     
                    
    Theta_zh = store['Theta_zh'].values
    Psi_sz = store['Psi_sz'].values    
    true_mem_size = store['Dts'].values.shape[1]    
    hyper2id = dict(store['hyper2id'].values)
    obj2id = dict(store['source2id'].values)    
    trace_fpath = store['trace_fpath'][0][0]
    
    #SEQ_FILE_PATH = createTestingSeqFile(store)
    testDic = {}
    testSetCount = 0
    r = open(SEQ_FILE_PATH, 'r')    
    for line in r:
        line = line.strip() 
        tmp = line.split('\t')
        user = tmp[0]      
        if(user not in hyper2id):
            print("User: "+str(user)+" is not found in training set !")
            continue
        testSetCount += 1
        if(user in testDic):
            testDic[user].append(line)                                                    
        else:
            testDic[user]=[line]
                        
    r.close()
        
   
    smoothedProbs = {}
    if(UNBIAS_CATS_WITH_FREQ):
        print('>>> calculating statistics for unbiasing categories ...')
        smoothedProbs = calculatingItemsFreq(trace_fpath, true_mem_size)
        
    
    print('Number of test samples: '+str(testSetCount))   
    myProcs = []
    idealCoreQuota = testSetCount // CORES
    userList = testDic.keys()    
    uid = 0
    q = Queue()
    for i in range(CORES):  
        coreTestDic = {}
        coreShare = 0
        while uid < len(userList):
            coreShare += len(testDic[userList[uid]])
            coreTestDic[userList[uid]] = testDic[userList[uid]]
            uid += 1
            if(coreShare >= idealCoreQuota):
                p = Process(target=outlierDetection, args=(coreTestDic, coreShare, i, q , store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, smoothedProbs))
                #outlierDetection(coreTestDic, coreShare, i, q , store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, smoothedProbs)
                myProcs.append(p)         
                testSetCount -= coreShare
                leftCores = (CORES-(i+1))
                if(leftCores >0):
                    idealCoreQuota = testSetCount // leftCores 
                print('>>> Starting process: '+str(i)+' on '+str(coreShare)+' samples.')
                p.start()       
                break
                                    
        myProcs.append(p)        
        
        
        
    for i in range(CORES):
        myProcs[i].join()
        print('>>> process: '+str(i)+' finished')
    
    
    #results = []
    #for i in range(CORES):
    #    results.append(q.get(True))
            
                            
    print('\n>>> All DONE!')
    store.close()

                                       
def main():   
    distributeOutlierDetection() 
  
    

if __name__ == "__main__":
    main()    
    #cProfile.run('distributeOutlierDetection()')
    #plac.call(main)
    print('DONE!')
