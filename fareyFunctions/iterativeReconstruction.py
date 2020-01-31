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
import tensorflow as tf
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

Assumes K-Space is not centered
'''
def abmlem_frt_complex(iterations, N, g_j, mValues, BU, BL, smoothingNum, smoothIncrement, oversampleFilter=None, epsilon=0.001, h=None):
    smoothMidIteration = iterations // 2
    smoothMaxIteration = iterations - iterations // 10
    center = False
    # Generate first estimate of the image
    f = finite.ifrt_complex(g_j, N, mValues=mValues, center=center, oversampleFilter=oversampleFilter)
    fPrev = np.array(f, copy=True)
    fOld = np.zeros_like(f, dtype=np.complex)
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
        f = smoothing(f, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i, h=h)
        
    # Done backprojecting    
    print("Finished performing ABfMLEM")
    return f, fPrev


'''
iterations is the maximum number of iterations
N is the dimension of the output image
g_j is the sinogram
mValues are the gradients
t is the tuning number

Assumes the frt of certain mValues returns 0s for other mValues

Assumes K-Space is not centered
'''
def sirt_frt_complex(iterations, N, g_j, mValues, t, smoothingNum, smoothIncrement, oversampleFilter=None, h=None):
    smoothMidIteration = iterations // 2
    smoothMaxIteration = iterations - iterations // 10
    center = False
    # Generate first estimate of the image
    f = finite.ifrt_complex(g_j, N, mValues=mValues, center=center, oversampleFilter=oversampleFilter)
    fPrev = np.array(f, copy=True)
    fOld = np.zeros_like(f, dtype=np.complex)
    
    # Get number of projection data
    [numProj, lenProj] = g_j.shape
    mu = numProj * lenProj
    mu = len(mValues)
    
    for i in range(0, iterations):
        print("Iteration: ", i)
            
        # Get previous image
        fOld = f
        
        # Numerator (error term)
        num = g_j - finite.frt_complex(fOld, N, mValues=mValues, center=center)
        
        # Obtain first image
        f = fOld + t * finite.ifrt_complex(num / mu, N, mValues=mValues, center=center, oversampleFilter=oversampleFilter)
        
        # Smooth image
        f = smoothing(f, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i, h=h)
        
    # Done backprojecting    
    print("Finished performing SIRT")
    return f, fPrev

'''
Performs SIRT reconstruction directly in k-space

Assumes K-Space is not centered
'''
def sirt_fft_complex(iterations, N, F, fractal, t, smoothingNum, smoothIncrement, h=None):
    smoothMidIteration = iterations // 2
    smoothMaxIteration = iterations - iterations // 8
    
    # Generate first estimate of the image
    f = fftpack.ifft2(F)
    fPrev = np.array(f, copy=True)
    fOld = np.zeros_like(f, dtype=np.complex)
    
    for i in range(0, iterations):
        print("Iteration: ", i)
        
        # Get previous image
        fOld = f
        
        # Numerator (error term)
        a = fftpack.fft2(fOld)
        num = (F - a) * fractal
        
        # Obtain first image
        f = fOld + t * fftpack.ifft2(num / N**2)
        
        # Smooth image
        f = smoothing(f, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i, h=h)
        
    # Done backprojecting    
    print("Finished performing FFTSIRT")
    return f, fPrev

'''
Performs SIRT reconstruction directly in k-space using TF

Assumes K-Space is not centred

N - Number of pixels on the side of image
F - k-space
FStep - Current predicted image k-space
fStep - Current predicted image
fractal - shape of fractal in k-space
t - Constant which controls rate

Returns image and k-space
'''
# @tf.function
def tf_sirt_fft_complex(iterations, N, F, fractal, t, smoothingNum, smoothIncrement, centered=False, h=None):
    
    smoothMidIteration = iterations // 2
    smoothMaxIteration = iterations - iterations // 8
    
     
    Ftf = tf.convert_to_tensor(F, dtype=tf.complex128)
    
    firstRecon = tf.signal.ifft2d(Ftf)
    
    fractaltf = tf.convert_to_tensor(fractal, dtype=tf.complex128)
    # Generate first estimate of the image
    f = tf.signal.ifft2d(Ftf)
    fOld = tf.zeros_like(f, dtype=tf.complex128)
    
    for i in tf.range(0, iterations):
        
        # Get next iteration's prediction
        fOld = tf_sirt_fft_complex_helper(N, Ftf, fOld, fractaltf, t)
        
        # Smooth image
        fOld = smoothing(fOld, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i, h=h)
        
    return fOld, firstRecon
    
@tf.function
def tf_sirt_fft_complex_helper(N, F, fStep, fractal, t):
    
    # Get k-space of current iteration
    FStep = tf.signal.fft2d(fStep)
    
    # Determine the numerator (Error term)
    num = tf.math.multiply((F - FStep), fractal)    
    
    # Obtain image after SIRT step
    fNext = fStep + t * tf.signal.ifft2d(num / N**2)
    
    # Here we want to smooth the image in a tensorflow manner
    
    # Return predicted image
    return fNext

'''
Performs SIRT reconstruction directly in k-space

Assumes K-Space is not centered
'''
# Not really working
def sirt_fft_complex_multi(iterations, N, F, fractal, t, smoothingNum, smoothIncrement, centered=False, complexOutput=False, h=None):
    smoothMidIteration = iterations // 2
    smoothMaxIteration = iterations - iterations // 8
        
    # Store partial images
    f = np.zeros_like(F, np.complex)
    a = np.zeros_like(F, np.complex)
    
    # Generate first estimate of the image
    for i, ch in enumerate(F):
        f[i, :, :] = fftpack.ifft2(ch)
        
    # Normalize the data
    fMax = np.max(np.abs(f))
    f = f * 255 / fMax
    
    for i, im in enumerate(f):
        F[i, :, :] = fftpack.fft2(im)
    
    fPrev = np.array(f, copy=True)
    fOld = np.zeros_like(f, dtype=np.complex)
    
    for i in range(0, iterations):
        print("Iteration: ", i)
                   
        # Get previous image
        fOld = np.array(f, copy=True)
        
        # Numerator (error term)
        for imNum, im in enumerate(fOld):
            a[imNum, :, :] = fftpack.fft2(im)
        num = (F - a) * fractal
        
        # Obtain first image
        for imNum, im in enumerate(fOld):
            f[imNum, :, :] = im + t * fftpack.ifft2(num[imNum, :, :] / N**2)
        
        
        # Smooth image
        for imNum, im in enumerate(f):
            f[imNum, :, :] = smoothing(im, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i, h=h, k=True)
            
            
    # Done backprojecting    
    print("Finished performing multi-channel FFTSIRT")
        
    newImage = np.zeros((N, N), dtype=np.complex)
    # Combine the data from each channel
    for im in f:
        newImage += (im ** 2)

    newImage = np.sqrt(newImage)
    if not complexOutput:
        newImage = np.absolute(newImage)
    newImage = newImage * 255 / np.max(np.abs(newImage))
    return newImage, fPrev

'''
Smooths reconstructed image
'''
def smoothing(image, smoothReconMode, smoothIncrement, smoothMidIteration, smoothMaxIteration, iterationNumber, h=None, k=None):
    f = image
    fdtype = float
    if smoothReconMode > 0 and iterationNumber % smoothIncrement == 0 and iterationNumber > 0: #smooth to stem growth of noise
            if k:
                fCenter = fftpack.fftshift(image) #avoid padding issues with some smoothing algorithms by ensuring image is centered
            else:
                fCenter = image
            fReal = np.real(fCenter)
            fImag = np.imag(fCenter)
            if smoothReconMode == 1:
                print("Smooth TV")
                if not h:
                    h = 8
                if iterationNumber > smoothMidIteration and iterationNumber <= smoothMaxIteration:
                    h /= 2.0
                elif iterationNumber > smoothMaxIteration:
                    h /= 4.0
                fReal = denoise_tv_chambolle(fReal, h, multichannel=False)
                fImag = denoise_tv_chambolle(fImag, h, multichannel=False)
            elif smoothReconMode == 2:
                print("Smooth Median")
                fReal = ndimage.median_filter(fReal, 3).astype(fdtype)
                fImag = ndimage.median_filter(fImag, 3).astype(fdtype)
            elif smoothReconMode == 3:
                if not h:
                    h = 10
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
            if k:
                f = fftpack.ifftshift(f)
            else:
                f = f
    return f
