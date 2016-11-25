'''
Created on Nov 21, 2016

@author: zahran
'''
import pandas as pd
import random

MODEL_PATH = '/home/zahran/Desktop/shareFolder/PARSED_pins_repins_win10_noop_NoLeaveOut_pinterest.h5'
DATA_GEN = '/home/zahran/Desktop/shareFolder/generated_data'

burstCount = 5






def main():   
    store = pd.HDFStore(MODEL_PATH)     
                    
    Theta_zh = store['Theta_zh'].values
    Psi_sz = store['Psi_sz'].values    
    true_mem_size = store['Dts'].values.shape[1]    
    hyper2id = dict(store['hyper2id'].values)
    obj2id = dict(store['source2id'].values)    
    trace_fpath = store['trace_fpath'][0][0]
  
    

if __name__ == "__main__":
    main()       
    print('DONE!')