from cython import boundscheck, wraparound
cdef extern from "<math.h>" nogil:
    double log10(double x)
    double pow(double x, double y)



@boundscheck(False)
cpdef double getLogProb(double[:] logProbs, int listLen) nogil:
    cdef double pw = 0.0
    if(listLen == 1):
        return logProbs[0]
    
    pw = (-1)*logProbs[0] + getLogProb(logProbs[1:], (listLen-1))
    return logProbs[0]+log10(1+pow(10,pw))

@boundscheck(False)
cpdef double evaluate(int userId, int[:] history, int historyLen, int targetObjId, double[:, ::1] Theta_zh, double[:, ::1] Psi_sz, int env) nogil:        
    cdef double mem_factor = 1.0    
    cdef double candidateProb = 0.0    
    cdef int j = 0
    for j in xrange(historyLen):#for all B
        #i.e. multiply all psi[objid1,z]*psi[objid2,z]*..psi[objidB,z]
        mem_factor *= Psi_sz[history[j], env] # Psi[objId, env z]            
    #mem_factor *= 1.0 / (1 - Psi_sz[history[len(history)-1], env])# 1-Psi_sz[mem[B-1],z] == 1-psi_sz[objIdB,z]       
    candidateProb += mem_factor * Psi_sz[targetObjId, env] * Theta_zh[env, userId]                                              
    return log10(candidateProb)

@boundscheck(False)
cpdef double calculateSequenceProb(int[:] theSequence, int theSequenceLen, double[:] logSeqProbZ, int true_mem_size, int userId, double[:, ::1] Theta_zh, double[:, ::1] Psi_sz) nogil:                     
    cdef double seqProb = 0.0   
    cdef double seqProbZ = 1.0    
    cdef int targetObjId = -1
    cdef double prior = 0.0
    cdef double candProb = 0.0
    cdef int window = 0   
    #cdef int[:] history = np.zeros(theSequenceLen, dtype='i4')
    cdef int[:] history
    cdef int historyLen = 0
    cdef int targetObjIdx, z, i = 0
    cdef int wmax = 0
   
    window = min(true_mem_size, theSequenceLen)
    for z in xrange(Psi_sz.shape[1]): #for envs
        seqProbZ = 0.0        
        for targetObjIdx in range(0,theSequenceLen): #targetObjIdx=0 cannot be predicted we have to skip it
            if(targetObjIdx == 0):                
                targetObjId = theSequence[targetObjIdx]                            
                prior = Psi_sz[targetObjId, z]
                seqProbZ += log10(prior)
            else:                                                                            
                targetObjId = theSequence[targetObjIdx]      
                wmax = max(0,targetObjIdx-window)           
                history = theSequence[wmax: targetObjIdx] # look back 'window' actions.                                                            
                historyLen = targetObjIdx-wmax                            
                candProb = evaluate(userId, history, historyLen, targetObjId, Theta_zh, Psi_sz, z) #(int[:, ::1] HOs, double[:, ::1] Theta_zh, double[:, ::1] Psi_sz, int[::1] count_z, int env):                                
                seqProbZ += candProb
        logSeqProbZ[z] = seqProbZ             
        #seqProb += seqProbZ
    #return logSeqProbZ     
    return getLogProb(logSeqProbZ, Psi_sz.shape[1])    
