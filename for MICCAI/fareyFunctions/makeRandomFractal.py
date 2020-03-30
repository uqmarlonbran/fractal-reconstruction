# -*- coding: utf-8 -*-
"""
Compute the CS reconstruction via the NLCG, Lustig's algorithm

Created on Fri Nov  9 09:26:46 2018

@author: shakes
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './finitetransform')
# from __future__ import print_function    # (at top of module)
# import _libpath #add custom libs
import finitetransform.farey as farey #local module
import finitetransform.radon as radon
import numpy as np
import finite
import time
import math

#cs
import param_class
import cs_utils
import scipy.fftpack as fftpack
import pyfftw
#import scipy.io as io

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

def makeRandomFractal(N, R, withTiling=False, tilingSize=8, centered=False, test=False):
    #parameters
    k = 1
    if R >= 6:
        K = 0.35
    else:
        K = 0.1    
    twoQuads = True
    
    fareyVectors = farey.Farey()        
    fareyVectors.compactOn()
    fareyVectors.generateFiniteWithCoverage(N)
    
    #sort to reorder result for prettier printing
    finiteAnglesSorted, anglesSorted = fareyVectors.sort('length')
    powSpect = np.zeros((N,N))
    #-------------------------------
    #compute finite lines
    lines, angles, mValues = finite.computeRandomLines(powSpect, anglesSorted, finiteAnglesSorted, R, K, centered, True)
    mu = len(lines)
    print("Number of finite lines:", len(lines))
    # print("Number of finite points:", len(lines)*(M-1))
    
    #sampling mask
    sampling_mask = np.zeros((N,N), np.float)
    for line in lines:
        u, v = line
        for x, y in zip(u, v):
            sampling_mask[x, y] += 1
    #determine oversampling because of power of two size
    #this is fixed for choice of M and m values
    oversamplingFilter = np.zeros((N,N), np.float)
    onesSlice = np.ones(N, np.float)
    for m in mValues:
        radon.setSlice(m, oversamplingFilter, onesSlice, 2)
    oversamplingFilter[oversamplingFilter==0] = 1
    sampling_mask /= oversamplingFilter
    sampling_mask = fftpack.fftshift(sampling_mask)
    count=0
    if withTiling:
        #tile center region further
        radius = N/tilingSize
        centerX = N/2
        centerY = N/2
        for i, row in enumerate(sampling_mask):
            for j, col in enumerate(row):
                distance = math.sqrt( (i-float(centerX))**2 + (j-float(centerY))**2)
                if distance < radius:
                    if not sampling_mask[i, j] > 0: #already selected
                        count += 1
                        sampling_mask[i, j] = 1
    totalSamples = mu*(N-1)+count+1
    actualR = float(totalSamples/N**2)
    print("Number of total sampled points:", totalSamples)
    print("Actual Reduction factor:", actualR)
    
    if not centered:
        sampling_mask = fftpack.ifftshift(sampling_mask)
        # oversamplingFilter = fftpack.ifftshift(oversamplingFilter)
    if not test:
        return sampling_mask, oversamplingFilter
    else:
        return sampling_mask, oversamplingFilter, mValues, lines