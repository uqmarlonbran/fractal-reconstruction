# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:44:05 2019

@author: uqmbran
"""

'''
This script performs MLEM reconstruction adapted from Shakes' implementation
'''
import pyfftw
import scipy.fftpack as fftpack
import numpy as np
import finite
from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, denoise_bilateral

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()


'''
iterations is the maximum number of iterations
N is the dimension of the output image
g_j is the sinogram
mValues are the gradients
epsilon is the stopping criteria

Assumes the frt of certain mValues returns 0s for other mValues
'''
def abmlem_frt_complex(iterations, N, g_j, mValues, BU, BL, smoothingNum, smoothIncrement, oversampleFilter=None, epsilon=0.001):
    smoothMidIteration = iterations // 2
    smoothMaxIteration = iterations - iterations // 10
    center = False
    # Generate first estimate of the image
    f = finite.ifrt_complex(g_j, N, mValues=mValues, center=center, oversampleFilter=oversampleFilter)
    fPrev = f
    fOld = np.zeros_like(f, dtype=np.complex)
    crit = 100
    BU = np.full((N, N), BU, dtype=np.complex)
    BL = np.full((N, N), BL, dtype=np.complex)
    
    # Get number of projection data
    [numProj, lenProj] = g_j.shape
    mu = numProj * lenProj
    mu = len(mValues)

    # Determine regularizing sinograms
    gL = g_j - finite.frt_complex(BL, N, center=center, mValues=mValues)
    gU = finite.frt_complex(BU, N, center=center, mValues=mValues) - g_j
    
    a = np.zeros_like(gU, dtype=np.complex)
    
    for i in range(0, iterations):
        print("Iteration: ", i)
        # Determine the stop criteria value
        if i != 0:
            crit = np.sqrt(np.sum(np.abs(f - fOld)) / np.sum(np.abs(f)))
            
        # Check if adequate image has been produced
        if crit >= epsilon or True:
            
            # Get previous image
            fOld = f
            
            # Get reused expressions
            L = fOld - BL
            U = BU - fOld
            
            # Project upper and lower images into DRT space
            fgL = finite.frt_complex(L, N, center=center, mValues=mValues)
            fgU = finite.frt_complex(U, N, center=center, mValues=mValues)
                     
            # Determine IL 
            a = np.divide(gL, fgL, where=fgL!=0)
            IL = L * finite.ifrt_complex(a, N, center=center, mValues=mValues, oversampleFilter=oversampleFilter) / mu
            
            # Determine IU
            a = np.divide(gU, fgU, where=fgU!=0)
            IU = U * finite.ifrt_complex(a, N, center=center, mValues=mValues, oversampleFilter=oversampleFilter) / mu
            
            # Get total image
            f = (IL * BU + IU * BL) / (IL + IU)
            
            # Smooth iamge
            f = smoothing(f, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i)
        
        # Done backprojecting    
        else:
            print("Finished")
            break
    return f, fPrev


'''
iterations is the maximum number of iterations
N is the dimension of the output image
g_j is the sinogram
mValues are the gradients
t is the tuning number

Assumes the frt of certain mValues returns 0s for other mValues
'''
def sirt_frt_complex(iterations, N, g_j, mValues, t, smoothingNum, smoothIncrement, oversampleFilter=None):
    smoothMidIteration = iterations // 2
    smoothMaxIteration = iterations - iterations // 10
    center = False
    # Generate first estimate of the image
    f = finite.ifrt_complex(g_j, N, mValues=mValues, center=center, oversampleFilter=oversampleFilter)
    fPrev = f
    fOld = np.zeros_like(f, dtype=np.complex)
    crit = 100
    epsilon = 1
    
    # Get number of projection data
    [numProj, lenProj] = g_j.shape
    mu = numProj * lenProj
    mu = len(mValues)
    
    for i in range(0, iterations):
        print("Iteration: ", i)
        # Determine the stop criteria value
        if i != 0:
            crit = np.sqrt(np.sum(np.abs(f - fOld)) / np.sum(np.abs(f)))
            
        # Check if adequate image has been produced
        if crit >= epsilon or True:
            
            # Get previous image
            fOld = f
            
            # Numerator (error term)
            num = g_j - finite.frt_complex(fOld, N, mValues=mValues, center=center)
            
            # Obtain first image
            f = fOld + t * finite.ifrt_complex(num / mu, N, mValues=mValues, center=center, oversampleFilter=oversampleFilter)
            
            # Smooth image
            f = smoothing(f, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i)
        
        # Done backprojecting    
        else:
            print("Finished")
            break
    return f, fPrev

'''
Performs SIRT reconstruction directly in k-space
'''
# Not really working
def sirt_fft_complex(iterations, N, F, fractal, t, smoothingNum, smoothIncrement, centered=False):
    smoothMidIteration = iterations // 2
    smoothMaxIteration = iterations - iterations // 10
#    center = False
    
    # Generate first estimate of the image
    f = fftpack.ifft2(F)
    fPrev = f
    fOld = np.zeros_like(f, dtype=np.complex)
    crit = 100
    epsilon = 1
    
    for i in range(0, iterations):
        print("Iteration: ", i)
        # Determine the stop criteria value
        if i != 0:
            crit = np.sqrt(np.sum(np.abs(f - fOld)) / np.sum(np.abs(f)))
            
        # Check if adequate image has been produced
        if crit >= epsilon or True:
            
            # Get previous image
            fOld = f
            
            # Numerator (error term)
            a = fftpack.fft2(fOld)
            num = (F - a) * fractal
            
            # Obtain first image
            f = fOld + t * fftpack.ifft2(num / N**2)
            
            # Smooth image
            f = smoothing(f, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i)
        
        # Done backprojecting    
        else:
            print("Finished")
            break
    return f, fPrev

'''
Performs SIRT reconstruction directly in k-space
'''
# Not really working
def sirt_fft_complex_circle(iterations, N, F, fractal, t, smoothingNum, smoothIncrement, centered=False):
    smoothMidIteration = iterations // 3
    smoothMaxIteration = iterations - iterations // 6
    
    ### Define circle within which we can perform reconstruction ###
    radius = N // 2
    c0, c1 = np.ogrid[0:N, 0:N]
    reconstruction_circle = ((c0 - N // 2) ** 2
                             + (c1 - N // 2) ** 2)
    reconstruction_circle = reconstruction_circle <= radius ** 2
    
    
    # Generate first estimate of the image
    f = fftpack.ifft2(F)
    fPrev = f
    fOld = np.zeros_like(f, dtype=np.complex)
    crit = 100
    epsilon = 1
    
    for i in range(0, iterations):
        print("Iteration: ", i)
        # Determine the stop criteria value
        if i != 0:
            crit = np.sqrt(np.sum(np.abs(f - fOld)) / np.sum(np.abs(f)))
            
        # Check if adequate image has been produced
        if crit >= epsilon or True:
            
            # Get previous image
            fOld = f
            
            # Numerator (error term)
            a = fftpack.fft2(fOld)
            num = (F - a) * fractal
            
            # Obtain first image
            f = fOld + t * fftpack.ifft2(num / N**2)
            
            f[reconstruction_circle == False] = 0
            
            # Smooth image
            f = smoothing(f, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i)
            
            
        # Done backprojecting    
        else:
            print("Finished")
            break
    return f, fPrev

'''
Smooths reconstructed image
'''
def smoothing(image, smoothReconMode, smoothIncrement, smoothMidIteration, smoothMaxIteration, iterationNumber):
    f = image
    fdtype = float
    if smoothReconMode > 0 and iterationNumber % smoothIncrement == 0 and iterationNumber > 0: #smooth to stem growth of noise
            fCenter = image #avoid padding issues with some smoothing algorithms by ensuring image is centered
            fReal = np.real(fCenter)
            fImag = np.imag(fCenter)
            if smoothReconMode == 1:
                print("Smooth TV")
                h = 2
                if iterationNumber > smoothMidIteration and iterationNumber <= smoothMaxIteration:
                    h /= 2.0
                elif iterationNumber > smoothMaxIteration:
                    h /= 4.0
                fReal = denoise_tv_chambolle(fReal, h, multichannel=False)
                fImag = denoise_tv_chambolle(fImag, h, multichannel=False)
            elif smoothReconMode == 2:
                print("Smooth Median")
                fReal = ndimage.median_filter(fReal, 4).astype(fdtype)
                fImag = ndimage.median_filter(fImag, 4).astype(fdtype)
            elif smoothReconMode == 3:
                h = 4
                '''
                NLM Smoothing Notes:
                A higher h results in a smoother image, at the expense of blurring features. 
                For a Gaussian noise of standard deviation sigma, a rule of thumb is to choose the value of h to be sigma of slightly less.
                The image is padded using the reflect mode of skimage.util.pad before denoising.
                '''
                if iterationNumber > smoothMidIteration and iterationNumber <= smoothMaxIteration:
                    h /= 2.0
                elif iterationNumber > smoothMaxIteration:
                    h /= 4.0
                print("Smooth NL h:",h)
                fReal = denoise_nl_means(fReal, patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                fImag = denoise_nl_means(fImag, patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
            elif smoothReconMode == 4:
                print("Smooth Bilateral")
                fReal = denoise_bilateral(fReal, sigma_color=0.15, sigma_spatial=7, multichannel=False).astype(fdtype)
                fImag = denoise_bilateral(fImag, sigma_color=0.15, sigma_spatial=7, multichannel=False).astype(fdtype)
            f = fReal +1j*fImag
    return f