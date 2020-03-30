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
import skimage.metrics as metrics
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, denoise_bilateral, denoise_wavelet
from l0_smoothing.l0_gradient_minimization import l0_gradient_minimization_2d

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
def rss(data, dim=0):
    return np.sqrt(np.sum(data ** 2, dim))

def ss(data, dim=0):
    return np.sum(data ** 2, dim)

#  generate mask based on alpha
def generate_mask_alpha(size=[128,128], r_factor_designed=5.0, r_alpha=3, axis_undersample=1,
                        acs=3, seed=0, mute=0):
    # init
    mask = np.zeros(size)
    # acs                
    if axis_undersample == 0:
        mask[:(acs+1)//2,:]=1
        mask[-acs//2:,:]=1
    else:
        mask[:,:(acs+1)//2]=1
        mask[:,-acs//2:]=1
    if seed>=0:
        np.random.seed(seed)
    # get samples
    num_phase_encode = size[axis_undersample]
    num_phase_sampled = int(np.floor(num_phase_encode/r_factor_designed))
    phase_encodes = np.array(range(0, num_phase_encode))
    centres = np.array(range(num_phase_encode//2 - acs//2, num_phase_encode//2 + acs//2))
    phase_encodes = np.delete(phase_encodes, centres)
    # coordinate
    coordinate_normalized = np.array(phase_encodes)
    coordinate_normalized = np.abs(coordinate_normalized-num_phase_encode/2)/(num_phase_encode/2.0)
    prob_sample = coordinate_normalized**r_alpha
    prob_sample = prob_sample/sum(prob_sample)
    # sample
    print(centres)
    print(len(phase_encodes))
    index_sample = np.random.choice(phase_encodes, size=num_phase_sampled-acs, 
                                    replace=False, p=prob_sample)
    
    # sample                
    if axis_undersample == 0:
        mask[index_sample,:]=1
    else:
        mask[:,index_sample]=1

    

    # compute reduction
    r_factor = len(mask.flatten())/sum(mask.flatten())
    if not mute:
        print('gen mask size of {1} for R-factor={0:.4f}'.format(r_factor, mask.shape))
        print(num_phase_encode, num_phase_sampled, np.where(mask[0,:]))

    return mask, r_factor

def abmlem_frt_complex(iterations, N, g_j, mValues, BU, BL, smoothingNum, smoothIncrement, oversampleFilter=None, epsilon=0.001, h=None, metric=False, ground=None):
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

    ssimArr = []
    psnrArr = []
    
    for i in range(0, iterations):
        # print("Iteration: ", i)
            
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
        
        # Update metrics
        if metric:
            ssimArr.append(metrics.structural_similarity(np.abs(ground), np.abs(f), data_range=np.max(np.abs(ground)) - np.min(np.abs(ground))))
            psnrArr.append(metrics.peak_signal_noise_ratio(np.abs(ground), np.abs(f), data_range=np.max(np.abs(ground)) - np.min(np.abs(ground))))
        
    # Done backprojecting    
    print("Finished performing ABfMLEM")
    if not metric:
        return f, fPrev
    else:
        return f, fPrev, psnrArr, ssimArr


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
        fOld = np.array(f, copy=True)
        
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
def sirt_fft_complex_multi(iterations, N, F, fractal, t, smoothingNum, smoothIncrement, centered=False, complexOutput=False, h=None, lmd=None):
        
    # Store partial images
    f = np.zeros_like(F, np.complex)
    
    # Generate first estimate of the image
    for i, ch in enumerate(F):
        f[i, :, :] = fftpack.ifft2(ch)
        
    # Normalize the data
    fMax = np.max(np.abs(f))
    f = f * 255 / fMax
    
    for i, im in enumerate(f):
        F[i, :, :] = fftpack.fft2(im)
    
    fPrev = np.array(f, copy=True)
    
    singles = np.zeros_like(F)
    
    # Perform sirt on each of the coils
    for i in range(0, F.shape[0]):
        f[i, :, :], _ = sirt_fft_complex(iterations, N, F[i, :, :], fractal, t, smoothingNum, smoothIncrement, h=h, k=True, lmd=lmd)
        printStr = "Finished " + str(i + 1) + " out of " + str(F.shape[0]) + " images."
        print(printStr)
    
    # Done backprojecting    
    print("Finished performing multi-channel FFTSIRT")
    
    if np.isscalar(N):
        newImage = np.zeros((N, N), dtype=np.complex)
    else:
        newImage = np.zeros((N[0], N[1]), dtype=np.complex)
    # Combine the data from each channel
    newImage = rss(f)
    
    if not complexOutput:
        newImage = np.absolute(newImage)
    # newImage = newImage * 255 / np.max(np.abs(newImage))
    return newImage, fPrev

'''
Performs SIRT reconstruction directly in k-space

Assumes K-Space is not centered
'''
def sirt_fft_complex(iterations, N, F, fractal, t, smoothingNum, smoothIncrement, h=None, lmd=None, k=None, metric=False, ground=None, insertSamples=False):
    
    if k:
        smoothMidIteration = iterations - iterations // 8
        smoothMaxIteration = iterations - iterations // 10
    else:
        smoothMidIteration = iterations // 2
        smoothMaxIteration = iterations - iterations // 10
    
    # Generate first estimate of the image
    f = fftpack.ifft2(F)
    # f = np.zeros_like(F)
    fPrev = np.array(f, copy=True)
    fOld = np.zeros_like(f, dtype=np.complex)
    
    psnrArr = []
    ssimArr = []
    
    # antiFrac = np.ones_like(fractal)
    # antiFrac = antiFrac - fractal
    
    if not np.isscalar(N):
        varN = np.max(N)
    else:
        varN = N
    
    for i in range(0, iterations):
        # print("Iteration: ", i)
        
        # Get previous image
        fOld = np.array(f, copy=True)
        
        # Numerator (error term)
        a = fftpack.fft2(fOld)
        num = (F - a) * fractal
        
        # Obtain first image
        f = fOld + t * fftpack.ifft2(num / varN**2)
        # f = fftpack.ifft2(a * antiFrac + F)
        
        # Smooth image
        f = smoothing(f, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i, h=h, lmd=lmd, k=k)
        
        # Update metrics
        if metric:
            ssimArr.append(metrics.structural_similarity(np.abs(ground), np.abs(f), data_range=np.max(np.abs(ground)) - np.min(np.abs(ground))))
            psnrArr.append(metrics.peak_signal_noise_ratio(np.abs(ground), np.abs(f), data_range=np.max(np.abs(ground)) - np.min(np.abs(ground))))
            
    # Re-insert known samples    
    if insertSamples:
        f = insert_samples(iterations, varN, F, f, fractal, t)
    
    # Done backprojecting    
    print("Finished performing FFTSIRT")
    if not metric:
        return f, fPrev
    else:
        return f, fPrev, psnrArr, ssimArr

def insert_samples(iterations, N, F, f, fractal, t):
    
    # Copy f array
    fOld = np.array(f, copy=True)
    
    # Re-insert known samples        
    for i in range(0, iterations):
        
        num = (F - fftpack.fft2(fOld)) * fractal
        fOld = fOld + t * fftpack.ifft2(num / N**2)
    
    return fOld

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
def tf_sirt_fft_complex(iterations, N, F, fractal, t, smoothingNum, smoothIncrement, centered=False, h=None, k=None, lmd=None):

    # Varying levels of smoothing depending on deepness
    smoothMidIteration = iterations // 2
    smoothMaxIteration = iterations - iterations // 8
    
    # Convert to tensor for parallel execution during iteration
    Ftf = tf.convert_to_tensor(F, dtype=tf.complex128)
    fractaltf = tf.convert_to_tensor(fractal, dtype=tf.complex128)
    
    # First estimate based on input data
    firstRecon = tf.signal.ifft2d(Ftf)
      
    # Generate first estimate of the image
    f = tf.identity(firstRecon)
    
    for i in tf.range(0, iterations):
        # Get next iteration's prediction
        f = tf_sirt_fft_complex_helper(tf.constant(N, dtype=tf.complex128), tf.constant(Ftf, dtype=tf.complex128), f, tf.constant(fractaltf, dtype=tf.complex128), tf.constant(t, dtype=tf.complex128))
        # Smooth here until tensorflow implementation is working
        f = smoothing(f, smoothingNum, smoothIncrement, smoothMidIteration, smoothMaxIteration, i, h=h, k=k, lmd=lmd)
    
    f = insert_samples_tf(tf.constant(iterations), tf.constant(N, dtype=tf.complex128), tf.constant(Ftf, dtype=tf.complex128), f, tf.constant(fractaltf, dtype=tf.complex128), tf.constant(t, dtype=tf.complex128))
    
    return f, firstRecon


def insert_samples_tf(iterations, N, F, f, fractal, t):
    
    # Copy f array
    fOld = tf.identity(f)
    
    # Re-insert known samples        
    for i in tf.range(0, iterations):
        
        num = (F - tf.signal.fft2d(fOld)) * fractal
        fOld = fOld + t * tf.signal.ifft2d(num / N**2)
    
    return fOld


"""
Tensorflow implementation of fftsirt iteration, performs smoothing step too
"""
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


# @tf.function
def tf_sirt_fft_complex_multi(iterations, N, F, fractal, t, smoothingNum, smoothIncrement, centered=False, h=None, lmd=None, complexOutput=False):
    
    # Get number of channels
    numCh, n1, n2 = F.shape
    f = np.zeros_like(F)
    # Generate first estimate of the image
    for i, ch in enumerate(F):
        f[i, :, :] = fftpack.ifft2(ch)
        
    # Normalize the data
    fMax = np.max(np.abs(f))
    f = f * 255 / fMax
    
    for i, im in enumerate(f):
        F[i, :, :] = fftpack.fft2(im)

    c1, _ = tf_sirt_fft_complex(iterations, N, F[0, :, :], fractal, t, smoothingNum, smoothIncrement, centered=centered, h=h, k=True, lmd=lmd)

    c2, _ = tf_sirt_fft_complex(iterations, N, F[1, :, :], fractal, t, smoothingNum, smoothIncrement, centered=centered, h=h, k=True, lmd=lmd)

    c3, _ = tf_sirt_fft_complex(iterations, N, F[2, :, :], fractal, t, smoothingNum, smoothIncrement, centered=centered, h=h, k=True, lmd=lmd)

    c4, _ = tf_sirt_fft_complex(iterations, N, F[3, :, :], fractal, t, smoothingNum, smoothIncrement, centered=centered, h=h, k=True, lmd=lmd)

    c5, _ = tf_sirt_fft_complex(iterations, N, F[4, :, :], fractal, t, smoothingNum, smoothIncrement, centered=centered, h=h, k=True, lmd=lmd)

    c6, _ = tf_sirt_fft_complex(iterations, N, F[5, :, :], fractal, t, smoothingNum, smoothIncrement, centered=centered, h=h, k=True, lmd=lmd)

    c7, _ = tf_sirt_fft_complex(iterations, N, F[6, :, :], fractal, t, smoothingNum, smoothIncrement, centered=centered, h=h, k=True, lmd=lmd)

    c8, _ = tf_sirt_fft_complex(iterations, N, F[7, :, :], fractal, t, smoothingNum, smoothIncrement, centered=centered, h=h, k=True, lmd=lmd)
        
    return c1, c2, c3, c4, c5, c6, c7, c8


"""
Function uses ADAM optimizer to optimize for data consistency and TV
"""
def reconstruct_ADAM_TV(epochs, steps_per_epoch, N, F, fractal, alpha, tv_weight, centered=False):
    
    # Generate first prediction (zf_image)
    f = tf.signal.ifft2d(F)
    f = tf.expand_dims(f, 2)
    fHmR = tf.Variable(tf.math.real(f))
    fHmI = tf.Variable(tf.math.imag(f))
    # Perform gradient descent
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            fHmR, fHmI = train_step(N, fHmR, fHmI, F, fractal, alpha, tv_weight)
            print(".", end='')
        print("Train step: {}".format(step))
    
    f = tf.complex(fHmR, fHmI)
    
    return tf.squeeze(f)

@tf.function
def loss_TV_Grad_Real(fHmR):
    
    with tf.GradientTape() as tape:
        loss = tf.math.reduce_sum(tf.image.total_variation(fHmR))
    # dloss_df
    return tape.gradient(loss, fHmR)

@tf.function
def loss_TV_Grad_Imag(fHmI):
    
    with tf.GradientTape() as tape:
        loss = tf.math.reduce_sum(tf.image.total_variation(fHmI))
    # dloss_df
    return tape.gradient(loss, fHmI)

"""
Use ADAM to optimize the image
"""

def train_step(N, fHmR, fHmI, zf_kspace, fractal, alpha, tv_weight):
    
    # Get current image kspace
    FStep = tf.signal.fft2d(tf.complex(fHmR, fHmI))
    
    # Determine the numerator (Error term)
    num = tf.math.multiply((zf_kspace - FStep), fractal)    
    
    # Real SIRT component
    SIRT = alpha * tf.signal.ifft2d(num / N**2)
    # print(fHmR)
    # print(loss_TV_Grad_Real(fHmR))
    
    # Obtain image after SIRT step
    tvR = loss_TV_Grad_Real(fHmR)
    tvI = loss_TV_Grad_Imag(fHmI)
    
    tv = tf.ones_like(tvR) * tv_weight
    
    fHmR = fHmR + tf.math.real(SIRT) + tf.math.multiply(tv, tvR)
    fHmI = fHmI + tf.math.imag(SIRT) + tf.math.multiply(tv, tvI)
                                                                
    return fHmR, fHmI
    
# """
# Function uses ADAM optimizer to optimize for data consistency and TV
# """
# def reconstruct_ADAM_TV(epochs, steps_per_epoch, F, fractal, alpha, tv_weight, centered=False):
    
#     # Generate first prediction (zf_image)
#     f = tf.signal.ifft2d(F)
    
    
#     fReal = tf.math.real(f)
#     fImag = tf.math.imag(f)
    
#     image = tf.Variable(tf.stack([fReal, fImag]))
    
#     # Create the optimizer
#     opt = tf.keras.optimizers.Adam()
    
#     # Perform gradient descent
#     step = 0
#     for n in range(epochs):
#         for m in range(steps_per_epoch):
#             step += 1
#             train_step(image, F, fractal, opt, alpha, tv_weight)
#             print(".", end='')
#         print("Train step: {}".format(step))
    
#     return tf.complex(image[0,:,:], image[1,:,:])

# """
# Attempting to write a loss function for ADAM optimizer

# Smooths image using TV
# """
# @tf.function
# def loss_fftSIRT(image, F, fractal, alpha, tv_weight):
    
#     f = tf.complex(image[0,:,:], image[1,:,:])
#     # Adds the number of channels for TV function
#     f = tf.expand_dims(f, 2)
    
#     # Get k-space of current iteration
#     Fit = tf.signal.fft2d(f)
    
#     # Determine the numerator (Error term)
#     tv = tf.math.reduce_sum(tf.image.total_variation(f))
#     sirt = tf.math.abs(tf.math.reduce_sum(tf.math.square(tf.math.multiply((F - Fit), fractal))))
#     sirt = tf.cast(sirt, tf.float32)
#     loss = alpha * sirt + tv_weight * tv
    
#     # Return loss
#     return loss

# """
# Use ADAM to optimize the image
# """
# @tf.function
# def train_step(image, zf_kspace, fractal, opt, alpha, tv_weight):
#     with tf.GradientTape() as tape:
#         loss = loss_fftSIRT(image, zf_kspace, fractal, alpha, tv_weight)
    
#     grad = tape.gradient(loss, image)
#     opt.apply_gradients([(grad, image)])
    
'''
Smooths reconstructed image
'''
def smoothing(image, smoothReconMode, smoothIncrement, smoothMidIteration, smoothMaxIteration, iterationNumber, h=None, lmd=None, k=None):
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
                fmax = np.max(np.abs(f))
                fmin = np.min(np.abs(f))
                f = (f - fmin) / fmax
                fReal = np.real(f)
                fImag = np.imag(f)
                fReal = denoise_bilateral(fReal, sigma_color=0.15, sigma_spatial=7, multichannel=False).astype(fdtype)
                fImag = denoise_bilateral(fImag, sigma_color=0.15, sigma_spatial=7, multichannel=False).astype(fdtype)
                fReal = fReal * fmax - fmin
                fImag = fImag * fmax - fmin
            elif smoothReconMode == 5:
                print("denoise wavelet")
                fReal = denoise_wavelet(fReal)
                fImag = denoise_wavelet(fImag)
            elif smoothReconMode == 6:
                lamd = lmd[0]
                beta_max = lmd[1]
                beta_rate = lmd[2]
                if not lmd:
                    lamd = 0.001
                    beta_max = 1.0e5
                    beta_rate = 2
                if iterationNumber > smoothMidIteration and iterationNumber <= smoothMaxIteration:
                    lamd /= 8.0
                    # beta_max /= 100
                    # beta_rate /= 2
                elif iterationNumber > smoothMaxIteration:
                    lamd /= 16.0
                    # beta_max /= 1000
                    # beta_rate /= 2
                    
                print("l0 Gradient Minimization, lmd: " + str(lamd) + "br: " + str(beta_rate) + "bm: " + str(beta_max))
                # fReal = l0_gradient_minimization_2d(fReal, lmd, beta_max, beta_rate)
                # fImag = l0_gradient_minimization_2d(fImag, lmd, beta_max, beta_rate)
                fmax = np.max(np.abs(f))
                f = f / fmax
                fReal = l0_gradient_minimization_2d(f, lamd, beta_max, beta_rate, max_iter=10000) * fmax
                # fImag = l0_gradient_minimization_2d(fImag, lamd, beta_max, beta_rate, max_iter=10000) * fmax
                # fReal = fReal * 255 / np.max(np.abs(fReal))
                fImag = 0
            f = fReal + 1j*fImag
            
            if k:
                f = fftpack.ifftshift(f)
            else:
                f = f
            
    return f
