#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False
from __future__ import division, print_function
######ZAHRAN
from __future__ import print_function
import __builtin__
##############################

from cpython cimport bool
from cython.parallel cimport prange

from tribeflow cimport _learn
from tribeflow.kernels.base cimport Kernel
from tribeflow.mycollections.stamp_lists cimport StampLists

import numpy as np

cdef extern from 'math.h':
    double log(double) nogil



def reciprocal_rank(int[:, ::1] HOs, double[:, ::1] Theta_zh, double[:, ::1] Psi_sz, int[::1] count_z, int env):
    '''
    Computes the reciprocal rank of predictions. Parameter descriptions
    below consider a burst of size `B`.

    Parameters
    ----------
    Dts: with inter event times. Shape is (n_events, B)
    HOs: hyper node (users) with burst (objetcs). Shape is (n_events, B+1). The
    last element in each row is the true object.
    previous_stamps: time_stamp to compute inter event times
    Theta_zh, Psi_sz, count_z: outputs of tribelow
    kernel: inter event kernel
    return_probs: boolean indicating if we should return full probabilities
     (unormalized for each row).
    
    Returns
    -------

    An array with reciprocal ranks and another with probabilities.
    '''
        
    cdef double dt = 0
    cdef int h = 0
    cdef int s = 0
    cdef int real_o = 0
    cdef int candidate_o = 0
    cdef int last_o = 0

    cdef int z = 0
    cdef int ns = Psi_sz.shape[0]
    
    cdef int[::1] mem = np.zeros((HOs.shape[1]-1), dtype='i4') #mem.shape = (B,)
    cdef double[::1] mem_factor = np.zeros(Psi_sz.shape[1], dtype='d') #mem_factor.shape = (nz,)
    cdef double[::1] p = np.zeros(Psi_sz.shape[0], dtype='d')# p.shape = (no,)
    
    cdef double[:] rrs = np.zeros(shape=(HOs.shape[0], ), dtype='d')
    cdef double[:, ::1] predictions = np.zeros(shape=(HOs.shape[0], ns), dtype='d')

    cdef int i, j
    for i in xrange(HOs.shape[0]): # for all test instances        
        h = HOs[i, 0]
        for j in xrange(mem.shape[0]): # for B (excluding the last object which is tha to be predicted one)
            mem[j] = HOs[i, 1 + j] #mem will hold a list of the B history obj ids. i.e. mem[0] = id(ob1)
        real_o = HOs[i, HOs.shape[1] - 1] # the gold object id to be predicted 
        last_o = HOs[i, HOs.shape[1] - 2] # the last object id to be predicted (object #B) 
        
        for candidate_o in prange(ns, schedule='static', nogil=True): # for all objects (ns = #objs)
            p[candidate_o] = 0.0
        #calculation for loop        
        for j in xrange(mem.shape[0]):#for all B
            #i.e. multiply all psi[objid1,z]*psi[objid2,z]*..psi[objidB,z]
            mem_factor[env] *= Psi_sz[mem[j], env] # Psi[objId, env z]
        
        mem_factor[z] *= 1.0 / (1 - Psi_sz[mem[mem.shape[0] - 1], z])# 1-Psi_sz[mem[B-1],z] == 1-psi_sz[objIdB,z]
       
        for candidate_o in prange(ns, schedule='static', nogil=True):
            p[candidate_o] += mem_factor[env] * Psi_sz[candidate_o, env] * Theta_zh[env, h]            
            #__builtin__.print('asd')
            #mf = mem_factor[env]
            #ps = Psi_sz[candidate_o, env]
            #thz = Theta_zh[env, h]
            #return mf
            #print mem_factor[env], Psi_sz[candidate_o, env], Theta_zh[env, h]
            #p[candidate_o] += mem_factor[z] * Psi_sz[candidate_o, z] * Theta_zh[z, h] * kernel.pdf(dt, z, previous_stamps)
          
        return np.array(p)
