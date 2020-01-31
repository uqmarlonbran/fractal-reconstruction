# -*- coding: utf-8 -*-
"""
Tomography related functions for image reconstruction from analog projections.
'simple example implementation for standard parallel beam projection/back-projection routines'

Created on Sun Sep 21 12:28:23 2014

@author: shakes and andrew
"""
import numpy
import scipy
from scipy import ndimage
import scipy.fftpack as fftpack
import random
import math

def project(img, angles):
    '''
    Project image using continuous geometry. The image is rotated (via linear interpolation)
    and projected
    '''
    Ny,Nx = img.shape
    Na = len(angles)
    NN = Nx + Nx/4 + Nx/4
    imgRot = numpy.zeros((NN,NN),dtype=numpy.float32)
    proj = numpy.zeros((Na,NN), dtype=numpy.float32)
    aa = NN/2 - Nx/2
    bb = NN/2 + Nx/2
    for a in numpy.arange(Na):
        imgRot[...] = 0.0
        imgRot[aa:bb,aa:bb] += img
        imgRot = scipy.ndimage.rotate(imgRot,-angles[a],reshape=False)
        for row in numpy.arange(NN):
            proj[a,:] += imgRot[row,:]
            
    return proj
    
def projection(img, angle):
    '''
    Create a single projection of the image using continuous geometry. The image is rotated (via linear interpolation)
    and projected
    '''
    Ny,Nx = img.shape
    NN = Nx + Nx/4 + Nx/4
    imgRot = numpy.zeros((NN,NN),dtype=numpy.float32)
    proj = numpy.zeros(NN, dtype=numpy.float32)
    aa = NN/2 - Nx/2
    bb = NN/2 + Nx/2
    imgRot[...] = 0.0
    imgRot[aa:bb,aa:bb] += img
    imgRot = scipy.ndimage.rotate(imgRot,-angle,reshape=False)
    for row in numpy.arange(NN):
        proj[:] += imgRot[row,:]
            
    return proj
    
def sliceSamples(theta, b, N, fftShape):
    '''
    Generate the b points along slice of length N at angle theta of DFT space with shape.
    '''
    r, s = fftShape
    phi = math.pi/2.0-theta
    projLengthAdjacent = float(N)*math.cos(phi)
    projLengthOpposite = float(N)*math.sin(phi)
    #ensure origin is included
    u2 = numpy.linspace(0, -projLengthAdjacent/2.0, b/2.0, endpoint=False) + r/2.0
    u1 = numpy.linspace(projLengthAdjacent/2.0, 0, b/2.0, endpoint=False) + r/2.0
    u = numpy.concatenate((u1,u2),axis=0)
    v2 = numpy.linspace(0, projLengthOpposite/2.0, b/2.0, endpoint=False) + s/2.0
    v1 = numpy.linspace(-projLengthOpposite/2.0, 0, b/2.0, endpoint=False) + s/2.0
    v = numpy.concatenate((v1,v2),axis=0)
    #print "u:",u
    #print "v:",v
    return u, v
    
def randomSamples(row, b, fftShape):
    '''
    Generate random b points along row of DFT space with shape.
    '''
    r, s = fftShape
    u = numpy.full(b, row, dtype=numpy.int32)
    v = numpy.random.randint(0, high=s, size=b)
    #print "u:",u
    #print "v:",v
    return u, v
            
def ctft(img, padFactor=1):
    '''
    Continuous Time Fourier Transform of a image, assuming its band-limited.
    By default the padding to simulate the CTFT is minimal (by an addition N/2)
    This is the padFactor of 1, increase this to improve the approximation.
    '''
    Ny,Nx = img.shape
    NN = Nx + padFactor*Nx/4 + padFactor*Nx/4
    imgRot = numpy.zeros((NN,NN),dtype=numpy.float32)
    aa = NN/2 - Nx/2
    bb = NN/2 + Nx/2
    imgRot[aa:bb,aa:bb] += img
    fftImg = fftpack.fft2(imgRot) #the '2' is important
    return fftpack.fftshift(fftImg)

def addNoise( proj ):
    '''
    Add Gaussian noise to projections
    '''
    mu = 0.0
    sigma = 0.1*scipy.ndimage.standard_deviation(proj)
    Na,Nw = proj.shape
    for a in numpy.arange(Na):
      for w in numpy.arange(Nw):
        proj[a,w] += random.gauss(mu,sigma)

#reconstruction algorithms
#FBP
def rampFilter(proj, norm=True):
    '''
    Apply the ramp filter to the projections required for FBP
    This is done to convert polar to Cartesian coords.
    '''
    Na,Nw = proj.shape
    filt = numpy.abs( numpy.fft.fftfreq(Nw) )  
    filt = numpy.resize( filt , [ 1 + numpy.floor( 0.5*Nw ) ] )
    for a in numpy.arange(Na):
        temp = numpy.fft.rfft( proj[a,:] , Nw )
        temp *= filt
        proj[a,:] = numpy.fft.irfft( temp , Nw )
        if norm:
            proj[a,:] /= Nw
        
def rampFilterProjection(proj, norm=True):
    '''
    Apply the ramp filter to the projections required for FBP
    This is done to convert polar to Cartesian coords.
    '''
    Nw = proj.size
    filt = numpy.abs( numpy.fft.fftfreq(Nw) )  
    filt = numpy.resize( filt , [ 1 + numpy.floor( 0.5*Nw ) ] )
    temp = numpy.fft.rfft(proj , Nw)
    temp *= filt
    proj = numpy.fft.irfft( temp , Nw )
    if norm:
        proj /= Nw
    return proj
        
def rampFilterSlice(slice):
    '''
    Apply the ramp filter to the slice required for FBP.
    This is done to convert polar to Cartesian coords.
    '''
    Nw = len(slice)
    filt = numpy.abs( numpy.fft.fftfreq(Nw) )  
    filt = numpy.resize( filt , [ numpy.floor( Nw ) ] )
    slice *= filt
    return slice
    
def backProject(img, proj, angles, norm=True):
    '''
    Filtered backprojection (FBP) algorithm. Assumes filter has already been applied.
    Example:
    reconImg[...] = 0.0
    rampFilter(projectionData)
    backProject(reconImg,projectionData,projAngles)
    '''
    Na,Nw = proj.shape
    Ny,Nx = img.shape
    NN = Nx + Nx/4 + Nx/4
    imgRot = numpy.zeros((NN,NN),dtype=numpy.float32)
    aa = NN/2 - Nx/2
    bb = NN/2 + Nx/2
#    cc = NN/2 - Nw/2
#    dd = NN/2 + Nw/2
    for a in numpy.arange(Na):
        imgRot[...] = 0.0
        for row in numpy.arange(NN):
          imgRot[row,:] += proj[a,:]
        imgRot = scipy.ndimage.rotate(imgRot,angles[a],reshape=False)
        img += imgRot[aa:bb,aa:bb]
        
    if norm:
        img /= Na*Nw

#-------------
#helper functions
def getRadialCoordinates(theta, data, b=0):
    '''
    Get the radial slice coordinates u, v arrays (in pixel coordinates at angle theta) of a NxN discrete array using the Fourier slice theorem.
    This can be applied to the DFT arrays and is good for drawing sample points of the slice on plots.
    b is the number of sample points on the entire radial line.
    '''
    rows, cols = data.shape
    if b == 0:
        b = rows/2
    
    h = rows/(2.0*(b/2.0))
    uStep = h*math.cos(theta)
    vStep = h*math.sin(theta)
    
    
