#-*- coding: utf8
'''
Created on Oct 2, 2016

@author: zahran
'''

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

CORES = 2
ALPHA_NORANKING = 0.05
ALPHA_RANKING = 0.05

MODEL_PATH = '/home/zahran/Desktop/shareFolder/PARSED_pins_repins_win10_noop_NoLeaveOut_pinterest.h5'
SEQ_FILE_PATH = '/home/zahran/Desktop/shareFolder/sqlData_likes_full_info_fixed_ONLY_TRUE_friendship' 
#SEQ_FILE_PATH = '/home/zahran/Desktop/shareFolder/PARSED_pins_repins_win10_pinterest_INJECTED'
TESTSET_COUNT_ADJUST = True
UNBIAS_CATS_WITH_FREQ = False
smoothingParam = 1.0   #smootihng parameter for unbiasing item counts.
STAT_FILE = '/home/zahran/Desktop/shareFolder/Stats'

#def evaluate(HOs, Theta_zh, Psi_sz, count_z, env, candidate):  
def evaluate(userId, history, targetObjId, Theta_zh, Psi_sz, env):        
    mem_factor = 1.0    
    candidateProb = 0.0                    
    for j in xrange(len(history)):#for all B
        #i.e. multiply all psi[objid1,z]*psi[objid2,z]*..psi[objidB,z]
        mem_factor *= Psi_sz[history[j], env] # Psi[objId, env z]            
    #mem_factor *= 1.0 / (1 - Psi_sz[history[len(history)-1], env])# 1-Psi_sz[mem[B-1],z] == 1-psi_sz[objIdB,z]       
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

def bonferroni_hypothesis_testing(keySortedPvalues, pValues, pvalType, testsetlen):
    if(pvalType == 'RANKING'):
        ALPHA = ALPHA_RANKING
    else:
        ALPHA = ALPHA_NORANKING
    outlierVector = ['N/A']*len(keySortedPvalues)
    bonferroni_ALPHA = float(ALPHA)/float(len(keySortedPvalues))
    if(TESTSET_COUNT_ADJUST == True):
        bonferroni_ALPHA = bonferroni_ALPHA / float(testsetlen)
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

def holm_hypothesis_testing (keySortedPvalues, pValues, pvalType, testsetlen):
    if(pvalType == 'RANKING'):
        ALPHA = ALPHA_RANKING
    else:
        ALPHA = ALPHA_NORANKING
    k = -1
    outlierVector = ['N/A']*len(keySortedPvalues)  
    for i in range(len(keySortedPvalues)):
        if(TESTSET_COUNT_ADJUST == True):
            val = float(ALPHA)/float(((len(keySortedPvalues)*testsetlen)-i))
        else:            
            val = float(ALPHA)/float((len(keySortedPvalues)-i))
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
               
def updateChiSq(chiSqs, chiSqs_expected, decisions, friendship, dKey):
    
    for i in range(len(decisions)):        
        if(decisions[i] == 'OUTLIER' and friendship[i] == 'true'):
            chiSqs[dKey][0] += 1        
        elif(decisions[i] == 'OUTLIER' and friendship[i] == 'false'):
            chiSqs[dKey][1] += 1
        elif(decisions[i] == 'NORMAL' and friendship[i] == 'true'):
            chiSqs[dKey][2] += 1
        elif(decisions[i] == 'NORMAL' and friendship[i] == 'false'):
            chiSqs[dKey][3] += 1
            
    row0 = chiSqs[dKey][0] + chiSqs[dKey][1]
    row1 = chiSqs[dKey][2] + chiSqs[dKey][3]
    col0 = chiSqs[dKey][0] + chiSqs[dKey][2]
    col1 = chiSqs[dKey][1] + chiSqs[dKey][3]
    grandTotal = row0+row1
    
    
    chiSqs_expected[dKey][0] = float(row0*col0)/float(grandTotal)
    chiSqs_expected[dKey][1] = float(row0*col1)/float(grandTotal)
    chiSqs_expected[dKey][2] = float(row1*col0)/float(grandTotal)
    chiSqs_expected[dKey][3] = float(row1*col1)/float(grandTotal)
    
    chis = chisquare(chiSqs[dKey], f_exp=chiSqs_expected[dKey], ddof=2)
    return chis

def updateResultStats(resStats, decisions, injectionMarkers, dKey):
    #tp,fp,fn,tn
    for i in range(len(decisions)):   
        if(decisions[i] == 'OUTLIER' and injectionMarkers[i] == 'true'):
            resStats[dKey][0] += 1        
        elif(decisions[i] == 'OUTLIER' and injectionMarkers[i] == 'false'):
            resStats[dKey][1] += 1
        elif(decisions[i] == 'NORMAL' and injectionMarkers[i] == 'true'):
            resStats[dKey][2] += 1
        elif(decisions[i] == 'NORMAL' and injectionMarkers[i] == 'false'):
            resStats[dKey][3] += 1
    
    try:         
        rec  = float(resStats[dKey][0])/float(resStats[dKey][0] + resStats[dKey][2])
        prec = float(resStats[dKey][0])/float(resStats[dKey][0] + resStats[dKey][1])
        fscore= (2*prec*rec) / (prec+rec)
        return [rec, prec, fscore]
    except:
        return [0,0,0]
    
def outlierDetection_InjectionAnalysis(testLines, coreId, startLine, endLine, q, store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z, smoothedProbs):
    myCnt = 0
    mylog = open(SEQ_FILE_PATH+'_ANAOMLY_ANALYSIS_'+str(coreId),'w')    
    writer = open(SEQ_FILE_PATH+'_SCORES_ANAOMLY_ANALYSIS_'+str(coreId),'w')
    quota = endLine-startLine
    #tp,fp,fn,tn
    resStats = {'bon_rank':[0]*4, 'bon_noRank':[0]*4, 'holms_rank': [0]*4, 'holms_noRank':[0]*4}
    for t in range(startLine, endLine):
        tsLine = testLines[t]    
        myCnt += 1               
        tmp = tsLine.strip().split('\t') 
        user = tmp[0]
      
        if(user not in hyper2id):            
            mylog.write('User: '+str(user)+' is not found in trainingSet !\n')          
            writer.write('User: '+str(user)+' is not found in trainingSet !\n')
            continue
              
        seq = tmp[1:true_mem_size+2]
        injectionMarkers = tmp[true_mem_size+2:]            
                 
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
                    
        keySortedPvaluesWithRanks = sorted(pValuesWithRanks, key=lambda k: (-pValuesWithRanks[k], k), reverse=True)
        keySortedPvaluesWithoutRanks = sorted(pValuesWithoutRanks, key=lambda k: (-pValuesWithoutRanks[k], k), reverse=True)        

        outlierVector_bonferroniWithRanks = bonferroni_hypothesis_testing(keySortedPvaluesWithRanks, pValuesWithRanks, 'RANKING', len(testLines))
        outlierVector_bonferroniWithoutRanks = bonferroni_hypothesis_testing(keySortedPvaluesWithoutRanks, pValuesWithoutRanks, 'NORANKING', len(testLines))        
        outlierVector_holmsWithRanks = holm_hypothesis_testing(keySortedPvaluesWithRanks, pValuesWithRanks, 'RANKING', len(testLines))
        outlierVector_holmsWithoutRanks = holm_hypothesis_testing(keySortedPvaluesWithoutRanks, pValuesWithoutRanks, 'NORANKING', len(testLines))
        
        res_bon_rank = updateResultStats(resStats, outlierVector_bonferroniWithRanks, injectionMarkers, 'bon_rank')
        res_bon_noRank = updateResultStats(resStats, outlierVector_bonferroniWithoutRanks, injectionMarkers, 'bon_noRank')
        res_holms_rank = updateResultStats(resStats, outlierVector_holmsWithRanks, injectionMarkers, 'holms_rank')
        res_holms_noRank = updateResultStats(resStats, outlierVector_holmsWithoutRanks, injectionMarkers, 'holms_noRank')
        
        writer.write('PvaluesWithRanks##'+str(pValuesWithRanks)+'||PvaluesWithoutRanks##'+str(pValuesWithoutRanks)+'||goldMarkers##'+str(injectionMarkers)+'\n')
        mylog.write('userId: '+str(tmp[0])+'\n')
        mylog.write('Action pvalue_with_ranks bonferroni_withRanks holms_withRanks pValues_withoutRanks holms_withRanks bonferroni_withoutRanks holms_withoutRanks\n')
        for x in range(0,len(seq)):           
            mylog.write('||'+str(seq[x])+'|| INJECTION: '+str(injectionMarkers[x])+'|| ')                                                                
            mylog.write(str(pValuesWithRanks[x])+' ')
            mylog.write(str(outlierVector_bonferroniWithRanks[x])+' ')
            mylog.write(str(outlierVector_holmsWithRanks[x])+' ')
            mylog.write(str(pValuesWithoutRanks[x])+' ')
            mylog.write(str(outlierVector_bonferroniWithoutRanks[x])+' ')
            mylog.write(str(outlierVector_holmsWithoutRanks[x])+' ')
            
            #chiSqs = {'bon_rank':[0]*4, 'bon_noRank':[0]*4, 'holms_rank': [0]*4, 'holms_noRank':[0]*4}
            mylog.write('\n')                                                    
        
        mylog.write(str('\n'+'res_bon_rank: '+str(res_bon_rank)))
        mylog.write(str('\n'+'res_bon_norank: '+str(res_bon_noRank)))
        mylog.write(str('\n'+'res_holms_rank: '+str(res_holms_rank)))
        mylog.write(str('\n'+'res_holms_norank: '+str(res_holms_noRank)))
                
        mylog.write('\n\n')                        
        if(t%100 == 0):
            mylog.flush()
            writer.flush()
        print('>>> proc: '+ str(coreId)+' finished '+ str(myCnt)+'/'+str(quota)+' instances ...')            
    mylog.close()
    writer.close()
    #seqFile.close()
    ret = resStats
    q.put(ret)
  
def outlierDetection_FBanalysis(testLines, coreId, startLine, endLine, q, store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z, smoothedProbs):
    myCnt = 0
    mylog = open(SEQ_FILE_PATH+'_ANAOMLY_ANALYSIS_'+str(coreId),'w')
    writer = open(SEQ_FILE_PATH+'_SCORES_ANAOMLY_ANALYSIS_'+str(coreId),'w')
    
    chiSqs = {'bon_rank':[0]*4, 'bon_noRank':[0]*4, 'holms_rank': [0]*4, 'holms_noRank':[0]*4}
    chiSqs_expected = {'bon_rank':[0]*4, 'bon_noRank':[0]*4, 'holms_rank': [0]*4, 'holms_noRank':[0]*4}
    quota = endLine-startLine
    for t in range(startLine, endLine):
        tsLine = testLines[t]    
        myCnt += 1               
        tmp = tsLine.strip().split('\t') 
        user = tmp[0]
      
        if(user not in hyper2id):            
            mylog.write('User: '+str(user)+' is not found in trainingSet !\n')
            writer.write('User: '+str(user)+' is not found in trainingSet !\n')          
            continue
       
       
        seq = tmp[1:true_mem_size+2]
        frienship = tmp[true_mem_size+2:]            
        
         
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
                    
        keySortedPvaluesWithRanks = sorted(pValuesWithRanks, key=lambda k: (-pValuesWithRanks[k], k), reverse=True)
        keySortedPvaluesWithoutRanks = sorted(pValuesWithoutRanks, key=lambda k: (-pValuesWithoutRanks[k], k), reverse=True)

        outlierVector_bonferroniWithRanks = bonferroni_hypothesis_testing(keySortedPvaluesWithRanks, pValuesWithRanks, 'RANKING', len(testLines))
        outlierVector_bonferroniWithoutRanks = bonferroni_hypothesis_testing(keySortedPvaluesWithoutRanks, pValuesWithoutRanks, 'NORANKING', len(testLines))        
        outlierVector_holmsWithRanks = holm_hypothesis_testing(keySortedPvaluesWithRanks, pValuesWithRanks, 'RANKING', len(testLines))
        outlierVector_holmsWithoutRanks = holm_hypothesis_testing(keySortedPvaluesWithoutRanks, pValuesWithoutRanks, 'NORANKING', len(testLines))
        
        writer.write('PvaluesWithRanks##'+str(pValuesWithRanks)+'||PvaluesWithoutRanks##'+str(pValuesWithoutRanks)+'||goldMarkers##'+str(frienship)+'\n')
        mylog.write('userId: '+str(tmp[0])+'\n')
        mylog.write('Action pvalue_with_ranks bonferroni_withRanks holms_withRanks pValues_withoutRanks holms_withRanks bonferroni_withoutRanks holms_withoutRanks\n')
        for x in range(0,len(seq)):
            mylog.write('||'+str(seq[x])+'|| fb: '+str(frienship[x])+'|| ')                                
            mylog.write(str(pValuesWithRanks[x])+' ')
            mylog.write(str(outlierVector_bonferroniWithRanks[x])+' ')
            mylog.write(str(outlierVector_holmsWithRanks[x])+' ')
            mylog.write(str(pValuesWithoutRanks[x])+' ')
            mylog.write(str(outlierVector_bonferroniWithoutRanks[x])+' ')
            mylog.write(str(outlierVector_holmsWithoutRanks[x])+' ')                        
            mylog.write('\n')                                                    

        chi_bon_rank = updateChiSq(chiSqs, chiSqs_expected, outlierVector_bonferroniWithRanks, frienship, 'bon_rank')
        chi_bon_norank = updateChiSq(chiSqs, chiSqs_expected, outlierVector_bonferroniWithoutRanks, frienship, 'bon_noRank')
        chi_holms_rank = updateChiSq(chiSqs, chiSqs_expected, outlierVector_holmsWithRanks, frienship, 'holms_rank')
        chi_holms_norank = updateChiSq(chiSqs, chiSqs_expected, outlierVector_holmsWithoutRanks, frienship, 'holms_noRank')
                  
        mylog.write('\npredictionCounts: '+str(chiSqs))
        mylog.write('\nExpectedCounts:   '+str(chiSqs_expected))
        mylog.write(str('\n'+'chi_bon_rank: '+str(chi_bon_rank)))
        mylog.write(str('\n'+'chi_bon_norank: '+str(chi_bon_norank)))
        mylog.write(str('\n'+'chi_holms_rank: '+str(chi_holms_rank)))
        mylog.write(str('\n'+'chi_holms_norank: '+str(chi_holms_norank)))                    
        mylog.write('\n\n')                        
        if(t%100 == 0):
            mylog.flush()
            writer.close()
        print('>>> proc: '+ str(coreId)+' finished '+ str(myCnt)+'/'+str(quota)+' instances ...')            
    mylog.close()
    writer.close()
    #seqFile.close()
    ret = [chiSqs, chiSqs_expected]
    q.put(ret)                                          
                                            
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
                                                                                        
def doFbAnalysis():
    store = pd.HDFStore(MODEL_PATH)     
    
    
    #SEQ_FILE_PATH = createTestingSeqFile(store)
    r = open(SEQ_FILE_PATH, 'r')    
    testLines = r.readlines()    
    r.close()
                
       
    Theta_zh = store['Theta_zh'].values
    Psi_sz = store['Psi_sz'].values
    count_z = store['count_z'].values[:, 0]   
    true_mem_size = store['Dts'].values.shape[1]    
    hyper2id = dict(store['hyper2id'].values)
    obj2id = dict(store['source2id'].values)    
    trace_fpath = store['trace_fpath'][0][0]
   
    smoothedProbs = {}
    if(UNBIAS_CATS_WITH_FREQ):
        print('>>> calculating statistics for unbiasing categories ...')
        smoothedProbs = calculatingItemsFreq(trace_fpath, true_mem_size)
        
    
    myProcs = []
    coreQuota = len(testLines) // CORES
    
    q = Queue()
    for i in range(CORES):        
        startLine = i*coreQuota
        endLine = min((i+1)*coreQuota, len(testLines))
        p = Process(target=outlierDetection_FBanalysis, args=(testLines, i, startLine, endLine, q , store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z, smoothedProbs))
        myProcs.append(p)
        print('>>> Starting process: '+str(i)+' start: '+str(startLine)+' end: '+str(endLine))
        p.start()
        
        
    for i in range(CORES):
        myProcs[i].join()
        print('>>> process: '+str(i)+' finished')
    
    
    results = []
    for i in range(CORES):
        results.append(q.get(True))
        
    chiSqs = {'bon_rank':[0]*4, 'bon_noRank':[0]*4, 'holms_rank': [0]*4, 'holms_noRank':[0]*4}
    chiSqs_expected = {'bon_rank':[0]*4, 'bon_noRank':[0]*4, 'holms_rank': [0]*4, 'holms_noRank':[0]*4}
    techniques = chiSqs.keys()
    
    for res in results:
        curr_chisq = res[0]
        curr_chisq_exp = res[1]
        for key in techniques:            
            chiSqs[key] = np.add(chiSqs[key], curr_chisq[key])
            chiSqs_expected[key] = np.add(chiSqs_expected[key], curr_chisq_exp[key])
            
    
    w = open(SEQ_FILE_PATH+'_FINAL_ANALYSIS', 'w')
    for key in techniques:       
        chis = chisquare(chiSqs[key], f_exp=chiSqs_expected[key], ddof=2)
        w.write('\nMethod: '+str(key))
        w.write('\npredictionCounts: '+str(chiSqs[key]))
        w.write('\nExpectedCounts:   '+str(chiSqs_expected[key]))
        w.write(str('\n'+'Chi-Square: '+str(chis)))
        w.write('\n')
    w.close()
                            
    print('\n>>> All DONE!')
    store.close()

def doInjectionAnalysis():
    store = pd.HDFStore(MODEL_PATH)     

    r = open(SEQ_FILE_PATH, 'r')    
    testLines = r.readlines()    
    r.close()
                
       
    Theta_zh = store['Theta_zh'].values
    Psi_sz = store['Psi_sz'].values
    count_z = store['count_z'].values[:, 0]   
    true_mem_size = store['Dts'].values.shape[1]    
    hyper2id = dict(store['hyper2id'].values)
    obj2id = dict(store['source2id'].values)    
    trace_fpath = store['trace_fpath'][0][0]
   
    smoothedProbs = {}
    if(UNBIAS_CATS_WITH_FREQ):
        print('>>> calculating statistics for unbiasing categories ...')
        smoothedProbs = calculatingItemsFreq(trace_fpath, true_mem_size)
        
    
    myProcs = []
    coreQuota = len(testLines) // CORES
    #coreQuota = 8 // CORES
    
    q = Queue()
    for i in range(CORES):        
        startLine = i*coreQuota
        endLine = min((i+1)*coreQuota, len(testLines))
        #outlierDetection_InjectionAnalysis(testLines, i, startLine, endLine, q , store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z, smoothedProbs)
        p = Process(target=outlierDetection_InjectionAnalysis, args=(testLines, i, startLine, endLine, q , store, true_mem_size, hyper2id, obj2id, Theta_zh, Psi_sz, count_z, smoothedProbs))
        myProcs.append(p)
        print('>>> Starting process: '+str(i)+' start: '+str(startLine)+' end: '+str(endLine))
        p.start()
        
        
    for i in range(CORES):
        myProcs[i].join()
        print('>>> process: '+str(i)+' finished')
    
    
    results = []
    for i in range(CORES):
        results.append(q.get(True))   
    
    resultStat = {'bon_rank':[0]*4, 'bon_noRank':[0]*4, 'holms_rank': [0]*4, 'holms_noRank':[0]*4}    
    techniques = resultStat.keys()
    
    for res in results:            
        for key in techniques:            
            resultStat[key] = np.add(resultStat[key], res[key])
                    
    
    w = open(SEQ_FILE_PATH+'_FINAL_ANALYSIS', 'w')
    for key in techniques:               
        w.write('\nMethod: '+str(key))
        w.write('\ntp,fp,fn,tn: '+str(resultStat[key]))
        try:         
            rec  = float(resultStat[key][0])/float(resultStat[key][0] + resultStat[key][2])
            prec = float(resultStat[key][0])/float(resultStat[key][0] + resultStat[key][1])
            fscore= (2*prec*rec) / (prec+rec)
            w.write('\nrecall   : '+str(rec))
            w.write('\nprecision: '+str(prec))
            w.write('\nfscore   : '+str(fscore))
        except:
            w.write('\nrecall   : '+str(0))
            w.write('\nprecision: '+str(0))
            w.write('\nfscore   : '+str(0))       
        w.write('\n')
    w.close()
                            
    print('\n>>> All DONE!')
    store.close()
                                       
def main():   
    doFbAnalysis() 
    #doInjectionAnalysis()
  
    


main()    
#plac.call(main)
print('DONE!')
