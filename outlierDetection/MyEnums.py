'''
Created on Nov 23, 2016

@author: zahran
'''
from enum import Enum

class METRIC (Enum):
    CHI_SQUARE = 1
    REC_PREC_FSCORE = 2
    FISHER = 3

class TECHNIQUE(Enum):
    ALL_OR_NOTHING = 1
    ONE_IS_ENOUGH = 2
    MAJORITY_VOTING = 3
    
class PVALUE(Enum):
    WITH_RANKING = 1
    WITHOUT_RANKING = 2
    
    
class HYP(Enum):    
    BONFERRONI = 1
    HOLMS = 2
    
class DECISION (Enum):
    OUTLIER = 1
    NORMAL = 0
    UNDECIDED = -1
    
class GOLDMARKER (Enum):
    TRUE = 1
    FALSE = 0
    
class SEQ_PROB(Enum):
    TRIBEFLOW = 1
    NGRAM = 2
    RNNLM = 3

class USE_WINDOW (Enum):
    TRUE = 1
    FALSE = 2
    
class BOUNDARY (Enum):
    IGNORE = 1
    INCLUDE = 2 
