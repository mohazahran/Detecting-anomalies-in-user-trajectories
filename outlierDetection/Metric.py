'''
Created on Nov 23, 2016

@author: zahran
'''
from MyEnums import *
from scipy.stats import chisquare, fisher_exact
from scipy.stats.contingency import expected_freq, chi2_contingency
import numpy as np

class Metric:
    def __init__(self):
        self.type = None
        pass
    def update(self, decisions, goldMarkers):
        pass
    
    

class Chisq(Metric):    
    def __init__(self):  
        self.type = METRIC.CHI_SQUARE          
        self.OT = 0 #OT: Decision=outlier and friendship=True.
        self.OF = 0 
        self.NT = 0
        self.NF = 0 #NF: Decision=Normal  and friendship=False      
        self.expectedOT = 0
        self.expectedOF = 0
        self.expectedNT = 0
        self.expectedNF = 0
        self.stats = None
    
    def getSummary(self):
        myStr = 'OT='+str(self.OT)+', OF='+str(self.OF)+', NT='+str(self.NT)+', NF='+str(self.NF)+', stats='+str(self.stats)
        return myStr
        
    def update(self, decisions, goldMarkers):
        for i in range(len(decisions)):        
            if(decisions[i] == DECISION.OUTLIER and goldMarkers[i] == GOLDMARKER.TRUE):
                self.OT += 1        
            elif(decisions[i] == DECISION.OUTLIER and goldMarkers[i] == GOLDMARKER.FALSE):
                self.OF += 1
            elif(decisions[i] == DECISION.NORMAL and goldMarkers[i] == GOLDMARKER.TRUE):
                self.NT += 1
            elif(decisions[i] == DECISION.NORMAL and goldMarkers[i] == GOLDMARKER.FALSE):
                self.NF += 1
            
        row0 = self.OT + self.OF # no. of outliers
        row1 = self.NT + self.NF
        col0 = self.OT + self.NT
        col1 = self.OF + self.NF
        grandTotal = row0+row1
                
        self.expectedOT = float(row0*col0)/float(grandTotal)
        self.expectedOF = float(row0*col1)/float(grandTotal)
        self.expectedNT = float(row1*col0)/float(grandTotal)
        self.expectedNF = float(row1*col1)/float(grandTotal)
        
        self.stats = chisquare([self.OT, self.OF, self.NT, self.NF], f_exp=[self.expectedOT, self.expectedOF, self.expectedNT, self.expectedNF], ddof=2)
        #ci = chi2_contingency([self.OT, self.OF, self.NT, self.NF])
        
        #print(self.stats, oddsratio, pvalue)
        #print('myExpected:'+str([self.expectedOT, self.expectedOF, self.expectedNT, self.expectedNF]))
        #ep = expected_freq([self.OT, self.OF, self.NT, self.NF])
        #cm = np.array_equal(ep, np.array([self.expectedOT, self.expectedOF, self.expectedNT, self.expectedNF]))
        #print(cm)
        #if(not cm):       
        #    print('\nERROR in exp cnt\n')
        #print('\n')
        
        
        
        

class Fisher(Metric):    
    def __init__(self):  
        self.type = METRIC.FISHER          
        self.OT = 0 #OT: Decision=outlier and friendship=True.
        self.OF = 0 
        self.NT = 0
        self.NF = 0 #NF: Decision=Normal  and friendship=False             
        self.stats = None
    
    def getSummary(self):
        myStr = 'OT='+str(self.OT)+', OF='+str(self.OF)+', NT='+str(self.NT)+', NF='+str(self.NF)+', stats='+str(self.stats)
        return myStr
        
    def update(self, decisions, goldMarkers):
        for i in range(len(decisions)):        
            if(decisions[i] == DECISION.OUTLIER and goldMarkers[i] == GOLDMARKER.TRUE):
                self.OT += 1        
            elif(decisions[i] == DECISION.OUTLIER and goldMarkers[i] == GOLDMARKER.FALSE):
                self.OF += 1
            elif(decisions[i] == DECISION.NORMAL and goldMarkers[i] == GOLDMARKER.TRUE):
                self.NT += 1
            elif(decisions[i] == DECISION.NORMAL and goldMarkers[i] == GOLDMARKER.FALSE):
                self.NF += 1
            
        
        
        self.stats = [fisher_exact([[self.OT, self.OF], [self.NT, self.NF]])]
        
        
        
class rpf(Metric): #recall_precision_fscore
    def __init__(self): 
        self.type = METRIC.REC_PREC_FSCORE
        self.OT = 0 #OT: Decision=outlier and friendship=True. (tp)
        self.OF = 0 #fp
        self.NT = 0 #fn
        self.NF = 0 #NF: Decision=Normal  and friendship=False (tn)
        self.stats = None   
    
    def getSummary(self):
        myStr = 'OT='+str(self.OT)+', OF='+str(self.OF)+', NT='+str(self.NT)+', NF='+str(self.NF)+', stats='+str(self.stats)
        return myStr
    
    def update(self, decisions, goldMarkers):
        for i in range(len(decisions)):   
            if(decisions[i] == DECISION.OUTLIER and goldMarkers[i] == GOLDMARKER.TRUE):
                self.OT += 1        
            elif(decisions[i] == DECISION.OUTLIER and goldMarkers[i] == GOLDMARKER.FALSE):
                self.OF += 1
            elif(decisions[i] == DECISION.NORMAL and goldMarkers[i] == GOLDMARKER.TRUE):
                self.NT += 1
            elif(decisions[i] == DECISION.NORMAL and goldMarkers[i] == GOLDMARKER.FALSE):
                self.NF += 1
    
        try:         
            rec  = float(self.OT)/float(self.OT + self.NT) #tp/tp+fn
            prec = float(self.OT)/float(self.OT + self.OF)
            fscore= (2*prec*rec) / (prec+rec)
            self.stats = [rec, prec, fscore]
        except:
            self.stats = [0,0,0]
    
    
    
    
    
        