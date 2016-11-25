'''
Created on Oct 3, 2016

@author: zahran
'''
import pandas as pd
import random

MODEL_PATH = '/home/zahran/Desktop/shareFolder/PARSED_pins_repins_win10_noop_NoLeaveOut_pinterest.h5'
TRAINING_FILE = '/home/zahran/Desktop/shareFolder/PARSED_pins_repins_win10_pinterest'
INJECTED_TRAIN = '/home/zahran/Desktop/shareFolder/PARSED_pins_repins_win10_pinterest_INJECTED'

injectedInstancesCount = 1000000 # 0 for all the training data
doRandomization = False
maxInjections = 2


def main():
    store = pd.HDFStore(MODEL_PATH)     
    obj2id = dict(store['source2id'].values)
    allCats = obj2id.keys()
    Dts = store['Dts']
    winSize = Dts.shape[1]    
    w = open(INJECTED_TRAIN, 'w')        
    r = open(TRAINING_FILE, 'r')
    N = 0
    if(injectedInstancesCount == 0):        
        for l in r:
            N += 1
    else:           
        N = injectedInstancesCount
    r.close()
    r = open(TRAINING_FILE, 'r')
    #Reservoir Sampling Algorithm
    sample = []
    for i,line in enumerate(r):
        if i < N:
            sample.append(line)
        else:
            break
       
#     for i,line in enumerate(r):
#         if i < N:
#             sample.append(line)
#         elif i >= N and random.random() < N/float(i+1):
#             replace = random.randint(0,len(sample)-1)
#             sample[replace] = line
        #else:
        #    break
    print('injectedInstancesCount=',injectedInstancesCount)
    
    for line in sample:
        line = line.strip()
        parts = line.split('\t')
        times = parts[:winSize]
        user = parts[winSize]
        cats = parts[winSize+1:]
        markers = ['false']*(winSize+1)
        injectedIdx = random.sample(list(range(len(cats))), maxInjections)
        for idx in injectedIdx:
            originalCat = cats[idx]                                        
            if(idx == 0):                    
                while True:
                    randCat = random.sample(allCats, 1)[0]                        
                    ok = (randCat != originalCat) and (randCat != cats[idx+1])
                    if(ok):
                        break                            
            elif (idx == len(cats)-1):
                while True:
                    randCat = random.sample(allCats, 1)[0]                        
                    ok = (randCat != originalCat) and (randCat != cats[-2])
                    if(ok):
                        break
            else:
                while True:
                    randCat = random.sample(allCats, 1)[0]                        
                    ok = (randCat != originalCat) and (randCat != cats[idx+1]) and (randCat != cats[idx-1])
                    if(ok):
                        break
                    
            cats[idx]= randCat
            markers[idx] = 'true'
        
        w.write(user+'\t')
        for c in cats:
            w.write(c+'\t')
        for m in markers:
            w.write(m+'\t')
        w.write('\n')
    w.close()
    store.close()
    
                                    
            




main()
print('DONE !')