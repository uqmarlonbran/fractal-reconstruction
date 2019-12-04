# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:46:15 2019

@author: marlo
"""

import numpy as np
import math
import finite
import radon


# Generate Fractal
def farey_fractal(N, F, twoQuads=True, centered=False):
    '''
    Create finite fractal for image size N given Farey order F (creates square
    fractal)
    
    Returns lines, angles, mValues, fractal formed (as an image), R (reduction)
    '''
    # Generate farey_sequence of order N
    octAngles = np.array(farey_sequence(F), dtype=np.complex)
    
    # Get dimensions of farey vector collection
    [l] = octAngles.shape
    
    # Create array to store farey vectors in the half-plane
    angles = np.zeros(l*2, dtype=np.complex)
    for i in range(0, 2):
        if i == 0:
            angles[int(l*i):int(l*(i + 1))] = octAngles
        elif i == 1:
            angles[int(l*i):int(l*(i + 1))] = octAngles.imag + 1j*octAngles.real
    
    # Due to symmetry we will get more than one of the same vector
    angles = np.unique(angles)
    
    # Generate kSpace grid
    fractal = np.zeros((N, N))
    
    # Compute lines and mValues
    lines, mValues = finite.computeLines(fractal, angles)
    mValues = np.unique(mValues)
    # Compute extra lines for mValues > N
    lines, angles, mValues = finite.computeKatzLines(fractal, angles, mValues, 60, twoQuads=twoQuads, centered=centered)
    lines = np.array(lines)
    # Obtain unique lines and mValues
    newM = []
    indexStore = []
    for i, m in enumerate(mValues):
        if m not in newM:
            newM.append(m)
            indexStore.append(i)
    mValues = newM
    lines = lines[indexStore,:,:].tolist()
    
    mu = len(lines)
    print("Number of finite lines in fractal: ", mu)
    
    # Paint fractal onto kSpace grid
    for line in lines:
        u, v = line
        for x, y in zip(u, v):
            fractal[x, y] = 1    
    
    R = np.sum(np.sum(fractal))/(N*N)
    
    # This has the number of times a point is sampled
    oversamplingFilter = np.zeros((N,N), np.complex)
    onesSlice = np.ones(N, np.uint32)
    for m in mValues:
            radon.setSlice(m, oversamplingFilter, onesSlice, 2)
    oversamplingFilter[oversamplingFilter==0] = 1
    
    return lines, angles, mValues, fractal, R, oversamplingFilter
   
def fullySampledFilter(N):
    if N%2==0:
        p = N + int(N/2)
    else:
        p = N + 1
    oversamplingFilter = np.zeros((N,N), np.complex)
    onesSlice = np.ones(N, np.uint32)
    for m in range(0, p):
            radon.setSlice(m, oversamplingFilter, onesSlice, 2)
    oversamplingFilter[oversamplingFilter==0] = 1
    return oversamplingFilter

# [b, a] or [y, x]
def farey_sequence(F):
    '''
    Returns Farey vectors as complex numbers
    '''
    # Initialise the list of Farey vectors
    angles = [complex(1, 0), complex(1, 1)]
    # Determine the total number of Farey components
    numPoints = farey_length(F)
    # While the number of elements is less than calculated number
    while (len(angles) < numPoints):
        # Add the next Farey sequence
        angles += farey_sequencer(angles, F)
        # Sort based on magnitude
        angles.sort(key=mag)
    
    return angles
        
# Get decimal size of vector for sorting
def mag(angle):
    '''
    Used for sorting the array of angles for next iteration
    '''
    b = angle.real
    a = angle.imag
    return a/b
      
# Determine the mediant fractions between each of the values in existing
def farey_sequencer(angles, F):
    newVals = []
    # Calculate the mediant value
    for x in range(1, len(angles)):
        b3 = angles[x-1].real + angles[x].real
        if b3 <= F:
            a3 = angles[x-1].imag + angles[x].imag
            newVals.append(complex(b3, a3))
    return newVals

# Number of elements in Farey sequence of order N
def farey_length(F):
    ''' 
    Calculates the number of elements in Farey sequence using Euler Totient
    Function
    
    Returns number of elements
    '''
    return 1 + sum(phi(f) for f in range(1, F + 1))
        
# A simple method to evaluate Euler Totient Function 
def phi(f): 
    result = 1
    for i in range(2, f): 
        if (math.gcd(i, f) == 1): 
            result = result + 1 
    return result

def normalizer(image):
    return (image - image.mean()) / image.std()