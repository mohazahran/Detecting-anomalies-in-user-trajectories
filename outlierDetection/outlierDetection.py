#-*- coding: utf8
'''
Created on Aug 9, 2016

@author: zahran
'''

#from __future__ import division, print_function
from scipy.stats import chisquare
from collections import OrderedDict
from multiprocessing import Process, Queue

import pandas as pd
#import plac
import numpy as np
import math
import os.path
from MyEnums import *
from TestSample import *
from bokeh.colors import gold
from DetectionTechnique import *

'''
CORES = 1
PATH = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/tribeflow_outlierDetection/pins_repins_fixedcat/'
RESULTS_PATH = PATH+'sim_Pvalues/'
MODEL_PATH = PATH+'pins_repins_win10_noop_NoLeaveOut.h5'
TRACE_PATH = PATH + 'pins_repins_win10.trace'
SEQ_FILE_PATH = PATH+'simData_perUser5'
STAT_FILE = PATH+'catStats'
UNBIAS_CATS_WITH_FREQ = False
HISTORY_SIZE = 4
smoothingParam = 1.0   #smootihng parameter for unbiasing item counts.
seq_prob = SEQ_PROB.NGRAM
useWindow = USE_WINDOW.FALSE
'''
#COMMON
CORES = 40
PATH = '/home/mohame11/pins_repins_fixedcat/'
RESULTS_PATH = PATH+'allLikes/pvalues_withWindow_log'
SEQ_FILE_PATH = PATH+'allLikes/likes.trace'
MODEL_PATH = PATH + 'pins_repins_win10_noop_NoLeaveOut.h5'
seq_prob = SEQ_PROB.TRIBEFLOW
useWindow = USE_WINDOW.TRUE

#TRIBEFLOW
TRACE_PATH = PATH + 'pins_repins_win10.trace'
STAT_FILE = PATH+'catStats'
UNBIAS_CATS_WITH_FREQ = True
smoothingParam = 1.0   #smoothing parameter for unbiasing item counts.

#NGRM/RNNLM
HISTORY_SIZE = 3
DATA_HAS_USER_INFO = False #has no effect on tribeflow
VARIABLE_SIZED_DATA = True #has no effect on tribeflow
ALL_ACTIONS_PATH = PATH + 'pins_repins_win4.trace_forLM_ALL_ACTIONS'

#RNNLM
RNNLM_PYTHON_PATH = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/rnnlm-python-master/scripts/'

                           
def getPvalueWithoutRanking(currentActionRank, keySortedProbs, probabilities):
    #normConst = 0.0
    #for i in range(len(probabilities)):
    #    normConst += probabilities[i]
        
    cdf = 0.0
    for i in range(currentActionRank+1):
        cdf += probabilities[keySortedProbs[i]]
    
    #prob = cdf/normConst
    return cdf
  
#testDic, quota, coreId, q, store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, smoothedProbs             
def get_norm_from_logScores(logScores):
    if(len(logScores) == 1):
        return logScores[0]
    pw = (-1)*logScores[0] + get_norm_from_logScores(logScores[1:])
    return logScores[0]+math.log10(1+math.pow(10,pw))


def outlierDetection(coreTestDic, quota, coreId, q, myModel):
    myCnt = 0    
    writer = open(RESULTS_PATH+'/outlier_analysis_pvalues_'+str(coreId),'w')
    
    for user in coreTestDic:
        for testSample in coreTestDic[user]:
            myCnt += 1
            #print(myCnt)
            seq = testSample.actions
            goldMarkers = testSample.goldMarkers
            #actions = myModel.obj2id.keys()    
            actions = myModel.getAllPossibleActions()              
            pValuesWithRanks = {}
            pValuesWithoutRanks = {}
            for i in range(len(seq)): #for all actions in the sequence.
                #Take the action with index i and replace it with all possible actions             
                probabilities = {}
                scores = {}        
                newSeq = list(seq)
                #currentActionId = myModel.obj2id[newSeq[i]] #current action id
                currentActionIndex = actions.index(newSeq[i])# the current action index in the action list.
                #cal scores (an un-normalized sequence prob in tribeflow)
                #normalizingConst = 0
                for j in range(len(actions)): #for all possible actions that can replace the current action
                    del newSeq[i]                
                    newSeq.insert(i, actions[j])    
                    userId = myModel.getUserId(user)     
                    if(myModel.type == SEQ_PROB.RNNLM):
                        seqScore = myModel.getProbability(userId, newSeq, coreId)
                    else:
                        seqScore = myModel.getProbability(userId, newSeq)  
                    scores[j] = seqScore
                    #normalizingConst += seqScore
                #cal probabilities                                                                                                                             
                #if(normalizingConst <= 1e-10000): #very small almost zero probability
                #    break
                logNormalizingConst = get_norm_from_logScores(scores.values())
                for j in range(len(actions)): #for all possible actions that can replace the current action
                    logProb = float(scores[j]) - float(logNormalizingConst)
                    probabilities[j] = math.pow(10, logProb)
                    #probabilities[j] = float(scores[j])/float(normalizingConst)
                #sorting ascendingly
                keySortedProbs = sorted(probabilities, key=lambda k: (-probabilities[k], k), reverse=True)
                currentActionRank = keySortedProbs.index(currentActionIndex)
                currentActionPvalueWithoutRanks = getPvalueWithoutRanking(currentActionRank, keySortedProbs, probabilities)
                currentActionPvalueWithRanks = float(currentActionRank+1)/float(len(actions))
                pValuesWithRanks[i] = currentActionPvalueWithRanks
                pValuesWithoutRanks[i] = currentActionPvalueWithoutRanks
            if(len(seq) == len(pValuesWithoutRanks)):                    
                writer.write('user##'+str(user)+'||seq##'+str(seq)+'||PvaluesWithRanks##'+str(pValuesWithRanks)+'||PvaluesWithoutRanks##'+str(pValuesWithoutRanks)+'||goldMarkers##'+str(goldMarkers)+'\n')        
            if(myCnt%100 == 0):
                writer.flush()
                print('>>> proc: '+ str(coreId)+' finished '+ str(myCnt)+'/'+str(quota)+' instances ...')                
    writer.close()    
    #ret = [chiSqs, chiSqs_expected]
    #q.put(ret)                                          
                                                                                                                                    
def distributeOutlierDetection():
    myModel = None
    
    if(seq_prob == SEQ_PROB.NGRAM):
        myModel = NgramLM()
        myModel.useWindow = useWindow
        myModel.model_path = MODEL_PATH
        myModel.true_mem_size = HISTORY_SIZE
        myModel.SEQ_FILE_PATH = SEQ_FILE_PATH
        myModel.DATA_HAS_USER_INFO = DATA_HAS_USER_INFO
        myModel.VARIABLE_SIZED_DATA = VARIABLE_SIZED_DATA
        myModel.ALL_ACTIONS_PATH=ALL_ACTIONS_PATH
        myModel.loadModel()
    
    elif(seq_prob == SEQ_PROB.RNNLM):
        myModel = RNNLM()
        myModel.useWindow = useWindow
        myModel.model_path = MODEL_PATH
        myModel.true_mem_size = HISTORY_SIZE
        myModel.SEQ_FILE_PATH = SEQ_FILE_PATH
        myModel.DATA_HAS_USER_INFO = DATA_HAS_USER_INFO
        myModel.VARIABLE_SIZED_DATA = VARIABLE_SIZED_DATA
        myModel.RNNLM_PYTHON_PATH = RNNLM_PYTHON_PATH
        myModel.RESULTS_PATH = RESULTS_PATH
        myModel.ALL_ACTIONS_PATH=ALL_ACTIONS_PATH
        myModel.loadModel()
    
    elif(seq_prob == SEQ_PROB.TRIBEFLOW):        
        myModel = TribeFlow()
        myModel.useWindow = useWindow
        
        myModel.model_path = MODEL_PATH
        myModel.store = pd.HDFStore(MODEL_PATH)
        myModel.Theta_zh = myModel.store['Theta_zh'].values
        myModel.Psi_sz = myModel.store['Psi_sz'].values    
        myModel.true_mem_size = myModel.store['Dts'].values.shape[1]    
        myModel.hyper2id = dict(myModel.store['hyper2id'].values)
        myModel.obj2id = dict(myModel.store['source2id'].values)    
        #myModel.trace_fpath = myModel.store['trace_fpath'][0][0]
        myModel.trace_fpath = TRACE_PATH
        myModel.UNBIAS_CATS_WITH_FREQ = UNBIAS_CATS_WITH_FREQ
        myModel.STAT_FILE = STAT_FILE
        myModel.SEQ_FILE_PATH=SEQ_FILE_PATH
        myModel.DATA_HAS_USER_INFO = DATA_HAS_USER_INFO
        myModel.VARIABLE_SIZED_DATA = VARIABLE_SIZED_DATA
 
        if(UNBIAS_CATS_WITH_FREQ):
            print('>>> calculating statistics for unbiasing categories ...')
            myModel.calculatingItemsFreq(smoothingParam)
    
    testDic,testSetCount = myModel.prepareTestSet()
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
                p = Process(target=outlierDetection, args=(coreTestDic, coreShare, i, q, myModel))
                #outlierDetection(coreTestDic, coreShare, i, q, myModel)
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
    #store.close()

                                       
def main():   
    distributeOutlierDetection() 
  
    

if __name__ == "__main__":
    main()    
    #cProfile.run('distributeOutlierDetection()')
    #plac.call(main)
    print('DONE!')
