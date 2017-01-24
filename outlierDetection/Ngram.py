'''
Created on Jan 21, 2017

@author: mohame11
'''
from MyEnums import *
from TestSample import *

class Ngram():
                              
    def formOriginalSeq(self, tests):
        origSeq = list(tests[0].actions)  
        #origGoldMarkers = list(tests[0].goldMarkers)
        for i in range(1,len(tests)):
            a = tests[i].actions[-1]
            #g = tests[i].goldMarkers[-1]
            origSeq.append(a)
            #origGoldMarkers.append(g)            
        return origSeq 

    def __init__(self, trainPath, isDataFormatted, useWindow, true_mem_size):
        w = open(trainPath+'_forLM', 'w')
        testDic = {}
        if(isDataFormatted == False): #neeed to clean the data to be just tokens
            r = open(trainPath, 'r')
            for line in r:
                line = line.strip() 
                tmp = line.split('\t')  
                user = tmp[true_mem_size]      
                seq = tmp[true_mem_size+1:]
                #goldMarkers = ['false']*len(seq)
                t = TestSample()  
                t.user = user
                t.actions = list(seq)
                #t.goldMarkers = list(goldMarkers)   
                
                if(user in testDic):
                    testDic[user].append(t)                                                    
                else:
                    testDic[user]=[t]
            r.close()
            if(useWindow == USE_WINDOW.FALSE): # we need to use the original sequence instead of overlapping windows
                for u in testDic:
                    tests = testDic[u]
                    originalSeq = self.formOriginalSeq(tests)
                    #self.data.append(originalSeq)
                    w.write(' '.join(originalSeq)+'\n')
                    w.flush()
            else:
                for u in testDic:
                    tests = testDic[u]
                    for t in tests:
                        #self.data.append(t.actions)
                        w.write(' '.join(originalSeq)+'\n')
        w.close()
    
    
def main():
    tracePath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/tribeflow_outlierDetection/pins_repins_fixedcat/pins_repins_win10.trace'
    isDataFormatted = False
    useWindow = USE_WINDOW.FALSE
    true_mem_size = 9
    ng = Ngram(tracePath, isDataFormatted, useWindow, true_mem_size)
    
    
    
main()
print('DONE!')    
                        
        