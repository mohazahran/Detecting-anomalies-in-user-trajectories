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

import numpy as np
import math
from os import listdir
from os.path import isfile, join
import outlierDetectionMulticore
import ast
import re



ANALYSIS_FILES_PATH = '/home/zahran/Desktop/shareFolder/'
FILE_NAME = 'PARSED_sqlData_likes_full_info_fixed_withFriendship_SCORES_ANAOMLY_ANALYSIS_'

def parseAnalysisFiles():
    retList = []
    pattern = re.compile(FILE_NAME+'\d+')
    allfiles = listdir(ANALYSIS_FILES_PATH)
    for file in allfiles:    
        if isfile(join(ANALYSIS_FILES_PATH, file)):
            
            if(pattern.match(file)):
            #if(FILE_NAME in file and '~' not in file):
                r = open(join(ANALYSIS_FILES_PATH, file), 'r')
                print(file)                                              
                for line in r:
                    if('User' in line):
                        continue
                    info = line.split('||')
                    dataList = []
                    for piece in info:                                           
                        data = piece.split('##')[1]
                        formatedContainer = ast.literal_eval(data)
                        dataList.append(formatedContainer)
                        
                    retList.append(dataList)
                           
    return retList


def bonferroni_hypothesis_testing(alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvalues, pValues, pvalType, testsetlen):
    if(pvalType == 'RANKING'):
        ALPHA = alphaRanking
    else:
        ALPHA = alphaNoRanking
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

def holm_hypothesis_testing (alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvalues, pValues, pvalType, testsetlen):
    if(pvalType == 'RANKING'):
        ALPHA = alphaRanking
    else:
        ALPHA = alphaNoRanking   
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

def doChiSqaure(allData, ALPHA_RANKING, ALPHA_NORANKING, TESTSET_COUNT_ADJUST):
    
    wRanking = open(ANALYSIS_FILES_PATH+FILE_NAME+'CHISQ_ALPHA_RANKING', 'w')
    wNoRanking = open(ANALYSIS_FILES_PATH+FILE_NAME+'CHISQ_ALPHA_NORANKING', 'w')
    
    minChiSqPvalue = [1000.0]*4
    minAlphas = [10000.0]*4
    
    wRanking.write('alpha, bon_rank_chiSqaure, bon_rank_chiSqaure\n')
    wNoRanking.write('alpha, bon_norank_chiSqaure, bon_norank_chiSqaure\n')
    
    for i in range(len(ALPHA_NORANKING)):
        print(i,'/',len(ALPHA_NORANKING))
        alphaRanking = ALPHA_RANKING[i]
        alphaNoRanking = ALPHA_NORANKING[i]
        chiSqs = {'bon_rank':[0]*4, 'bon_noRank':[0]*4, 'holms_rank': [0]*4, 'holms_noRank':[0]*4}
        chiSqs_expected = {'bon_rank':[0]*4, 'bon_noRank':[0]*4, 'holms_rank': [0]*4, 'holms_noRank':[0]*4}
        
        chi_bon_rank = None
        chi_bon_norank = None
        chi_holms_rank = None
        chi_holms_norank = None
        
        for inst in allData:
            
            pValuesWithRanks = inst[0]
            pValuesWithoutRanks = inst[1]
            goldMarkers = inst[2]
            
            keySortedPvaluesWithRanks = sorted(pValuesWithRanks, key=lambda k: (-pValuesWithRanks[k], k), reverse=True)
            keySortedPvaluesWithoutRanks = sorted(pValuesWithoutRanks, key=lambda k: (-pValuesWithoutRanks[k], k), reverse=True)
            
            #alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvalues, pValues, pvalType, testsetlen
            
            outlierVector_bonferroniWithRanks = bonferroni_hypothesis_testing \
            (alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvaluesWithRanks, pValuesWithRanks, 'RANKING', len(allData))
            
            outlierVector_bonferroniWithoutRanks = bonferroni_hypothesis_testing \
            (alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvaluesWithoutRanks, pValuesWithoutRanks, 'NORANKING', len(allData)) 
                   
            outlierVector_holmsWithRanks = holm_hypothesis_testing \
            (alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvaluesWithRanks, pValuesWithRanks, 'RANKING', len(allData))
            
            outlierVector_holmsWithoutRanks = holm_hypothesis_testing \
            (alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvaluesWithoutRanks, pValuesWithoutRanks, 'NORANKING', len(allData))
            
            
            chi_bon_rank = updateChiSq(chiSqs, chiSqs_expected, outlierVector_bonferroniWithRanks, goldMarkers, 'bon_rank')
            chi_bon_norank = updateChiSq(chiSqs, chiSqs_expected, outlierVector_bonferroniWithoutRanks, goldMarkers, 'bon_noRank')
            chi_holms_rank = updateChiSq(chiSqs, chiSqs_expected, outlierVector_holmsWithRanks, goldMarkers, 'holms_rank')
            chi_holms_norank = updateChiSq(chiSqs, chiSqs_expected, outlierVector_holmsWithoutRanks, goldMarkers, 'holms_noRank')
    
        chi_bon_rankVal = float(str(chi_bon_rank).split('=')[-1].replace(')',''))
        chi_holms_rankVal = float(str(chi_holms_rank).split('=')[-1].replace(')',''))
        chi_bon_norankVal = float(str(chi_bon_norank).split('=')[-1].replace(')',''))
        chi_holms_norankVal = float(str(chi_holms_norank).split('=')[-1].replace(')',''))
        
        wRanking.write(str(alphaRanking)+','+str(chi_bon_rankVal)+','+str(chi_holms_rankVal)+'\n')
        wNoRanking.write(str(alphaNoRanking)+','+str(chi_bon_norankVal)+','+str(chi_holms_norankVal)+'\n')
        
        if(minChiSqPvalue[0]>chi_bon_rankVal):
            minChiSqPvalue[0] = chi_bon_rankVal
            minAlphas[0] = alphaRanking
            
        if(minChiSqPvalue[1]>chi_bon_norankVal):
            minChiSqPvalue[1] = chi_bon_norankVal
            minAlphas[1] = alphaNoRanking
        
        if(minChiSqPvalue[2]>chi_holms_rankVal):
            minChiSqPvalue[2] = chi_holms_rankVal
            minAlphas[2] = alphaRanking
        
        if(minChiSqPvalue[3]>chi_holms_norankVal):
            minChiSqPvalue[3] = chi_holms_norankVal
            minAlphas[3] = alphaNoRanking
    
    wRanking.write('Min pvalue res_bon_rank  ='+str(minChiSqPvalue[0])+' alphaRank='+str(minAlphas[0])+'\n')
    wRanking.write('Min pvalue res_holms_rank='+str(minChiSqPvalue[2])+' alphaRank='+str(minAlphas[2])+'\n')
    
    wNoRanking.write('Min pvalue res_bon_noRank  ='+str(minChiSqPvalue[1])+' alphanoRank='+str(minAlphas[1])+'\n')
    wNoRanking.write('Min pvalue res_holms_noRank='+str(minChiSqPvalue[3])+' alphanoRank='+str(minAlphas[3])+'\n')
      
    wRanking.close()
    wNoRanking.close

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

def doRecallPrecFscore(allData, ALPHA_RANKING, ALPHA_NORANKING, TESTSET_COUNT_ADJUST):
    wRanking = open(ANALYSIS_FILES_PATH+FILE_NAME+'RecPrecFscore_ALPHA_RANKING', 'w')
    wNoRanking = open(ANALYSIS_FILES_PATH+FILE_NAME+'RecPrecFscore_ALPHA_NORANKING', 'w')
    
    wRanking.write('alpha,bon_rank[rec, prec, fscore], holms_rank[rec, prec, fscore]\n')
    wNoRanking.write('alpha,bon_norank[rec, prec, fscore], holms_norank[rec, prec, fscore]\n')
               
    maxFscores = [0.0]*4
    maxAlphas = [0.0]*4
    
    for i in range(len(ALPHA_NORANKING)):
        print(i,'/',len(ALPHA_NORANKING))
        alphaRanking = ALPHA_RANKING[i]
        alphaNoRanking = ALPHA_NORANKING[i]
        
        resStats = {'bon_rank':[0]*4, 'bon_noRank':[0]*4, 'holms_rank': [0]*4, 'holms_noRank':[0]*4}
        
        for inst in allData:
            
            pValuesWithRanks = inst[0]
            pValuesWithoutRanks = inst[1]
            goldMarkers = inst[2]
            
            keySortedPvaluesWithRanks = sorted(pValuesWithRanks, key=lambda k: (-pValuesWithRanks[k], k), reverse=True)
            keySortedPvaluesWithoutRanks = sorted(pValuesWithoutRanks, key=lambda k: (-pValuesWithoutRanks[k], k), reverse=True)
            
            #alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvalues, pValues, pvalType, testsetlen
            
            outlierVector_bonferroniWithRanks = bonferroni_hypothesis_testing \
            (alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvaluesWithRanks, pValuesWithRanks, 'RANKING', len(allData))
            
            outlierVector_bonferroniWithoutRanks = bonferroni_hypothesis_testing \
            (alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvaluesWithoutRanks, pValuesWithoutRanks, 'NORANKING', len(allData)) 
                   
            outlierVector_holmsWithRanks = holm_hypothesis_testing \
            (alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvaluesWithRanks, pValuesWithRanks, 'RANKING', len(allData))
            
            outlierVector_holmsWithoutRanks = holm_hypothesis_testing \
            (alphaRanking, alphaNoRanking, TESTSET_COUNT_ADJUST, keySortedPvaluesWithoutRanks, pValuesWithoutRanks, 'NORANKING', len(allData))
            
            
            res_bon_rank = updateResultStats(resStats, outlierVector_bonferroniWithRanks, goldMarkers, 'bon_rank')
            res_bon_noRank = updateResultStats(resStats, outlierVector_bonferroniWithoutRanks, goldMarkers, 'bon_noRank')
            res_holms_rank = updateResultStats(resStats, outlierVector_holmsWithRanks, goldMarkers, 'holms_rank')
            res_holms_noRank = updateResultStats(resStats, outlierVector_holmsWithoutRanks, goldMarkers, 'holms_noRank')
            
        if(maxFscores[0]<res_bon_rank[-1]):
            maxFscores[0] = res_bon_rank[-1]
            maxAlphas[0] = alphaRanking
            
        if(maxFscores[1]<res_bon_noRank[-1]):
            maxFscores[1] = res_bon_noRank[-1]
            maxAlphas[1] = alphaNoRanking
        
        if(maxFscores[2]<res_holms_rank[-1]):
            maxFscores[2] = res_holms_rank[-1]
            maxAlphas[2] = alphaRanking
        
        if(maxFscores[3]<res_holms_noRank[-1]):
            maxFscores[3] = res_holms_noRank[-1]
            maxAlphas[3] = alphaNoRanking
                
                
                
           
    
        
        wRanking.write(str(alphaRanking)+','+str(res_bon_rank)+','+str(res_holms_rank)+'\n')
        wNoRanking.write(str(alphaNoRanking)+','+str(res_bon_noRank)+','+str(res_holms_noRank)+'\n')
    
    wRanking.write('Max recall res_bon_rank  ='+str(maxFscores[0])+' alphaRank='+str(maxAlphas[0])+'\n')
    wRanking.write('Max recall res_holms_rank='+str(maxFscores[2])+' alphaRank='+str(maxAlphas[2])+'\n')
    
    wNoRanking.write('Max recall res_bon_noRank  ='+str(maxFscores[1])+' alphanoRank='+str(maxAlphas[1])+'\n')
    wNoRanking.write('Max recall res_holms_noRank='+str(maxFscores[3])+' alphanoRank='+str(maxAlphas[3])+'\n')
    
    wNoRanking.write('\n')
    
    wRanking.close()
    wNoRanking.close()
    
def main():
       
    ALPHA_NORANKING = np.arange(0.000005,0.1,0.005) # start=0, step=0.1, end=1 (exlusive)
    ALPHA_RANKING = np.arange(0.000005,0.1,0.005)    
    TESTSET_COUNT_ADJUST = True
    
    print('>>> Reading Data ...')
    allData = parseAnalysisFiles()
    print('>>> Evaluating ...')
    doChiSqaure(allData, ALPHA_RANKING, ALPHA_NORANKING, TESTSET_COUNT_ADJUST)
    #doRecallPrecFscore(allData, ALPHA_RANKING, ALPHA_NORANKING, TESTSET_COUNT_ADJUST)
  
    
main()    
print('DONE!')
