# -*- coding: utf-8 -*-
"""
Module for hyperfine imaging.

Created on Wed Feb 11 22:28:43 2015

@author: shakes
"""
import finitetransform.mojette as mojette
import finitetransform.radon as radon

def finiteAngles(angles, N):
    '''
    Return a list of finite angles for given angle set.
    '''
    mValues = []
    for index, angle in enumerate(angles):
        p, q = mojette.farey.get_pq(angle)
        
        m, inv = mojette.farey.toFinite(angle, N)
        mValues.append(m)
        
    return mValues
    
def sort(angles, finiteAngles):
    '''
    Sort list of angles and finite angles by finite number for N
    '''
    return [y for (y,x) in sorted(zip(finiteAngles, angles), key=lambda pair: pair[0])], [x for (y,x) in sorted(zip(finiteAngles, angles), key=lambda pair: pair[0])]

#retrieve members
import scipy.fftpack as fftpack
import numpy as np

def getMojetteProjections(dftSpace, angles, P, Q):
    '''
    Use the discrete, but non-periodic Fourier slice theorem to get corresponding Mojette transform (MT) projections.
    
    DFT space size N has to be chosen correctly (large enough), so that Mojette slices are Mojette projections.
    '''
    N, N = dftSpace.shape
    
    projections = []
    for angle in angles:
#        B = mojette.projectionLength(angle, P, Q)
        slice = radon.getMojetteSlice(angle, P, Q, dftSpace)
        projection = np.real(fftpack.ifft(slice))
        projections.append(projection)
        
    return projections
    
def getMojetteProjectionsFromSlices(dftSpace, finiteAngles, angles, P, Q):
    '''
    Use the discrete Fourier slice theorem to get corresponding discrete Radon transform (DRT) projections.
    
    If N is chosen correctly, DRT projections will Mojette projections.
    '''
    N, N = dftSpace.shape
    
    projections = []
    for m, angle in zip(finiteAngles, angles):
        slice = radon.getSlice(m, dftSpace)
        finiteProjection = np.real(fftpack.ifft(slice))
        projection = radon.mojetteProjection(finiteProjection, N, angle, P, Q)
        projections.append(projection)
        
    return projections
    
def measure(dftSpace, angles, P, Q, finite=False):
    '''
    Measure hyperfine projections from discrete Fourier space
    
    Convenience member. Same as getProjections()
    '''
    N, N = dftSpace.shape
    
    if not finite:
        return getMojetteProjections(dftSpace, angles, P, Q)
    else:
        finiteAnglesUsed = finiteAngles(angles, N)
        return getMojetteProjectionsFromSlices(dftSpace, finiteAnglesUsed, angles, P, Q)
    
#coordinates
import finitetransform.farey as farey
import math
    
def angleCoordinates(angle, b, N, center=False):
    '''
    Compute the 2D coordinates of each translate (in NxN DFT space) of farey angle of length b.
    b is the number of samples for each side of the DC
    N is assumed to be the larger space to embed lines
    Returns a list of u, v coordinate arrays [[u_0[...],v_0[...]], [u_1[...],v_1[...]], ...] for angle
    '''
    offset = 0.0
    if center:
        offset = N/2 #integer division deliberate
    
    u = []
    v = []
    
    #use mojette sampling directly
    p, q = farey.get_pq(angle)
    pFractional = p
    qFractional = q
#    print "pHat:", pFractional, "qHat:", qFractional

    #forward
    for translate in range(0, int(b)):
        translateP = N + pFractional*translate + offset 
        translateP %= N #prevent going outside padded area
        translateQ = N + qFractional*translate + offset 
        translateQ %= N #prevent going outside padded area
#        print "t:", translate, "tp:", translateP, "tq:", translateQ
        v.append( translateP )
        u.append( translateQ ) 
    #conjugate sym
#    print("Conj Sym:")
    for translate in range(0, int(b)):
        translateP = -pFractional*translate + offset 
        translateP %= N #prevent going outside padded area
        translateQ = -qFractional*translate + offset 
        translateQ %= N #prevent going outside padded area
#        print "t:", translate, "tp:", translateP, "tq:", translateQ
        v.append( translateP )
        u.append( translateQ ) 

    return u, v
    
def angleSliceCoordinates(angle, P, Q, N, center=False, symmetric=False, finite=False):
    '''
    Compute the 2D coordinates of each translate (in NxN DFT space) of every projection having angle.
    Returns a list of u, v coordinate arrays [[u_0[...],v_0[...]], [u_1[...],v_1[...]], ...] for angle
    '''
    offset = 0.0
    if center:
        offset = N/2 #integer division deliberate
    
    u = []
    v = []
    uSym = []
    vSym = []
    m, inv = farey.toFinite(angle, N)
    B = mojette.projectionLength(angle, P, Q)
    print("m:", m, "B:", B)
    
    if finite:
        #use finite method of generating the sampling
        for translate in range(0, B):
            translateFinite = (m*translate)%N #has issues in C, may need checking
            u.append( (translate+offset)%N )
            v.append( (translateFinite+offset)%N )
            if symmetric:
                uSym.append( (N-translate+offset)%N )
                vSym.append( (N-translateFinite+offset)%N )
    #        if translate > 50:
    #            break
    else:
        #use mojette sampling directly
        p, q = farey.get_pq(angle)
        pFractional = p*(N)/float(B)
        qFractional = q*(N)/float(B)
        print("pHat:", pFractional, "qHat:", qFractional)
        '''
        #2nd quad
        for translate in range(0, B):
            translateP = 4*(N-1) - pFractional*translate + offset #has issues in C, may need checking
            while translateP >= N-1:
                translateP -= N-1 #mod N
            print "t:", translate, "tp:", translateP
            translateQ = 4*(N-1) - qFractional*translate + offset #has issues in C, may need checking
            while translateQ >= N-1:
                translateQ -= N-1 #mod N
            u.append( translateP )
            v.append( translateQ )
        '''
        #2nd quad
        for translate in range(0, B):
            translateP = 4*N-pFractional*translate + offset #has issues in C, may need checking
            translateP = math.fmod(translateP, N)
            translateQ = 4*N+qFractional*translate + offset #has issues in C, may need checking
            translateQ = math.fmod(translateQ, N)
            print("t:", translate, "tp:", translateP, "tq:", translateQ)
            v.append( translateP )
            u.append( translateQ ) #DFT coordinates rotated by 90 degrees
        '''
        #3rd quad
        for translate in range(0, B):
            translateP = 4*(N-1) + pFractional*translate + offset #has issues in C, may need checking
            while translateP >= N-1:
                translateP -= N-1 #mod N
            print "t:", translate, "tp:", translateP
            translateQ = 4*(N-1) - qFractional*translate + offset #has issues in C, may need checking
            while translateQ >= N-1:
                translateQ -= N-1 #mod N
            u.append( translateQ )
            v.append( translateP )
        '''
        '''
        #4th quad
        for translate in range(0, B):
            translateP = 4*(N-1) - pFractional*translate + offset #has issues in C, may need checking
            while translateP >= N-1:
                translateP -= N-1 #mod N
            print "t:", translate, "tp:", translateP
            translateQ = 4*(N-1) - qFractional*translate + offset #has issues in C, may need checking
            while translateQ >= N-1:
                translateQ -= N-1 #mod N
            u.append( translateP )
            v.append( translateQ )
        '''
    
#    if symmetric:
#        uSym.reverse()
#        vSym.reverse()
#        u = np.concatenate((uSym, u))
#        v = np.concatenate((vSym, v))

    return u, v
