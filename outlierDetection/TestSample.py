'''
Created on Nov 23, 2016

@author: zahran
'''
from os import listdir
from os.path import isfile, join
import ast
import re
from MyEnums import *

class TestSample:
    def __init__(self):
        self.user = -1
        self.actions = []
        self.PvaluesWithRanks = {}
        self.PvaluesWithoutRanks = {}
        self.goldMarkers=[]
    
    @staticmethod
    def parseAnalysisFiles(FILE_NAME, ANALYSIS_FILES_PATH):
        user_test = {}
        pattern = re.compile(FILE_NAME+'\d+')
        allfiles = listdir(ANALYSIS_FILES_PATH)
        for file in allfiles:    
            if isfile(join(ANALYSIS_FILES_PATH, file)):            
                if(pattern.match(file) and '~' not in file):
                #if(FILE_NAME in file and '~' not in file):
                    r = open(join(ANALYSIS_FILES_PATH, file), 'r')
                    print(file)                                              
                    for line in r:
                        if('nan' in line):
                            #print('nan found !')
                            line = line.replace('nan', '1.0')                                     
                        info = line.split('||')
                        t = TestSample()                    
                        t.user = info[0].split('##')[1]
                        t.actions = ast.literal_eval(info[1].split('##')[1])
                        t.PvaluesWithRanks = ast.literal_eval(info[2].split('##')[1])
                        t.PvaluesWithoutRanks = ast.literal_eval(info[3].split('##')[1])
			#if(len(t.PvaluesWithRanks)<2 or len(t.PvaluesWithoutRanks)<2):
			#	print('>>>Bad line:',file,line)
                        golds = ast.literal_eval(info[4].split('##')[1])
                        for g in golds:
                            if(g == 'false'):
                                t.goldMarkers.append(GOLDMARKER.FALSE)
                            else:
                                t.goldMarkers.append(GOLDMARKER.TRUE)
                        if(t.user in user_test):
                            user_test[t.user].append(t)
                        else:
                            user_test[t.user] = [t]                       
                               
        return user_test
