# -*- coding: utf-8 -*-
"""
Module for computing all things related to ghosts.

@author: shakes
"""
import numpy as np
import carmichael as cm

def eigenvalues(N, fourier=True, returnOperators=False, proot=cm.PRIMITIVEROOT, modulus=cm.MODULUS):
    '''
    Compute the ghost eigenvalues for a given size N.
    Returns a 2D integer array of all possible eigenvalues. If returnOperators then (operators, eigenvalues) returned.
    '''
    operators = cm.zeros((N,N))
    eigenvalues = cm.zeros((N,N))
    
    shift = 0
    for op, eigen in zip(operators, eigenvalues): #!< Compute 0 <= m < p operators
    
        if returnOperators:
            #Setup Convolution Bases
            op[0] = cm.integer(1)
            if shift == N-1: #m = p operator
                op[1] = cm.negative_one(N, modulus)
            else:
                op[(shift+1)] = cm.negative_one(N, modulus)
        
        eigen[0] = cm.integer(1)
        if shift == N-1: #m = p operator
            eigen[1] = cm.negative_one(N, modulus)
        else:
            eigen[(shift+1)] = cm.negative_one(N, modulus)
        shift += 1

        #Find Eigenvalues of Operator
        eigen = cm.fct(eigen, N, cm.NTT_FORWARD, proot, modulus) #inplace
#        eigen = cm.ifct(eigen, N, proot, modulus)
#        print eigen
#        break
    
    if returnOperators:
        return operators, eigenvalues
    else:
        return eigenvalues

def convolutionOperator(N, eigenvals, shiftValues, returnEigenvalues=False, proot=cm.PRIMITIVEROOT, modulus=cm.MODULUS):
    '''
    Convolve ghosts in Carmichael/Fourier space from the list of shift values given.
    Eigenvalues is a 2D array ideally resulting from eigenvalues()
    Returns the 1D ghost convolution operator (in real space).
    '''
    #create delta function in Carmichael/Fourier space for convolution, should be all ones
    deltaFunc =  cm.ones(N)
    
    #Convolve ghosts
    for shift in shiftValues: #!< For each known projection (0 <= m < p)
        shift %= N #cyclic wrap shifts

        eigen = eigenvals[shift-1]
#        print "Eigenvalue:", eigen
        for index, value in enumerate(np.nditer(deltaFunc)): #!< For each bin
            deltaFunc[index] = (eigen[index] * value)%modulus
#            print bin

    funcEigenvalues = np.copy(deltaFunc)  
    funcEigenvalues = cm.fct(funcEigenvalues, N, cm.NTT_FORWARD, proot, modulus) #inplace
    funcEigenvalues /= N
        
    if returnEigenvalues:
        return funcEigenvalues, deltaFunc
    else:
        return deltaFunc
