# -*- coding: utf-8 -*-
"""
Load Knee images and test FFT SIRT reconstruction from the different sampling
schemes

@author: marlon
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './fareyFunctions')
import scipy.fftpack as fftpack
import pyfftw
import skimage.metrics as metrics
import loadNii
import numpy as np
import time
import math
import iterativeReconstruction as iterative
from fareyFractal import farey_fractal
import random
import matplotlib.pyplot as plt
from makeRandomFractal import makeRandomFractal
import finite
import param_class
import cs_utils
from fnlCg import fnlCg
from PIL import Image
from scipy.io import loadmat, savemat
import finitetransform.samplers as samplers
# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

#parameters
N = 320
floatType = np.complex
twoQuads = True
p = N

#-------------------------------
# Load fractals
print("Loading the fractals...")
# fractals, num_fractals = loadNii.load_data("knees/fractal")
# # Fractal indexes
# shakes = 1
# marlon = 0
# tiled = 2
# tiled2 = 3
def create_sampler(mode, M=256, r=0.5, r_alpha=2, acs=0., seed=-1, 
                       N=256, k=1, K=0.1, s=8, twoQuads=True):
    if 'cart' in mode.lower():
        params = {'M': M, 'r': r, 'r_alpha': r_alpha, 'acs': acs}
        if '1d' in mode.lower():
            sampler = samplers.OneDimCartesianRandomSampler(r=r, r_alpha=r_alpha, acs=acs, seed=-1)
        elif '2d' in mode.lower():
            sampler = samplers.TwoDimCartesianRandomSampler(r=r, r_alpha=r_alpha, acs=acs, seed=-1)
    elif 'fractal' in mode.lower():
        params = {'N': N, 'k': k, 'M': M, 'K': K, 'r': r, 's': s,
                'ctr': acs, 'twoQuads': twoQuads}
        sampler = samplers.FractalSampler(k, K, r, acs, s, seed=-1)
    return sampler, params
#-------------------------------
#load images data
# images, num_cases = loadNii.load_data("knees/knee_slices_reduced")

# load Cartesian data
dftSpace, num_cases = loadNii.load_data_channels("knees/knee_kspace_slices_reduced")
dftSpace = np.squeeze(dftSpace)
images = fftpack.fftshift(fftpack.ifft2(dftSpace))
dftSpace = np.zeros_like(images, dtype=np.complex)
imMax = np.max(np.abs(images))
#Attention: You must ensure the kspace data is correctly centered or not centered.
for i, im in enumerate(images):
    dftSpace[i, :, :] = fftpack.fft2(fftpack.ifftshift(im)) * 255 / imMax
    dftSpace[i, :, :] = np.roll(dftSpace[i, :, :], 1, axis=0)

images = fftpack.ifft2(dftSpace)

#Attention: You must ensure the kspace data is correctly centered or not centered.

dftSpace = fftpack.fft2(images)
dftSpace = dftSpace[0,:,:]
dftSpace = dftSpace / np.max(np.abs(dftSpace))
image = fftpack.ifftshift(fftpack.ifft2(dftSpace))
# dftSpace = fftpack.fftshift(fftpack.fft2(image * 255 / np.max(np.abs(image))))
dftSpace = fftpack.fftshift(fftpack.fft2(image))
image = fftpack.ifft2(dftSpace)
fftkSpaceShifted = dftSpace

im = Image.fromarray(np.rot90(np.abs(image*255/np.max(np.abs(image))), k=3).astype(np.uint8))
im.save("knee_lustig/knee" + '.png')

#compute lines
centered = False
#compute lines
fractal=True
random=True
cart2d=False
R = 4
r = [2, 4, 6, 8]
ssim = np.zeros((len(r)))
psnr = np.zeros_like(ssim)
elapsed = np.zeros_like(psnr)
rmse = np.zeros_like(psnr)
for j, R in enumerate(r):
    tilingSize=8
    if R == 2:
        fareyOrder = 10
        K = 2.4
        a=1.3
    elif R == 3:
        fareyOrder = 8
        K = 1.3
        tilingSize=8
    elif R == 4:
        fareyOrder = 7
        K = 0.88
        tilingSize=8
        a = 3.3
    elif R == 8:
        fareyOrder = 5
        K = 0.3
        tilingSize=8
        a = 10
    elif R == 10:
        tilingSize=13
    elif R == 6:
        a = 6
    
    # Generate the fractal
    # lines, angles, mValues, fractalMine, _, oversampleFilter = farey_fractal(N, fareyOrder, centered=centered)
    # Setup fractal
    if fractal:
        if not random:
            lines, angles, \
                mValues, sampling_mask, \
                oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', 
                                                    twoQuads=True)
            lines = np.array(lines)
        else:
            sampling_mask, samplingFilter = makeRandomFractal(N, (1/a), tilingSize=tilingSize, withTiling=True)
        
    
    #-------------------------------
    
    tilingSize=40
    if cart2d:
        sampler, params = create_sampler('cart_2d', M=N, r=R, acs=tilingSize, N=N)
        sampling_mask = fftpack.ifftshift(sampler.generate_mask((N,N)))
    elif not fractal:
        sampler, params = create_sampler('cart_1d', M=N, r=R, acs=tilingSize, N=N)
        sampling_mask = fftpack.ifftshift(sampler.generate_mask((N,N)))
    # print("Samples used: ", R)
    sampling_mask = fftpack.fftshift(sampling_mask)
    kSpace = fftkSpaceShifted*sampling_mask    
    
    #Define parameter class for nlcg
    params = param_class.nlcg_param()
    params.FTMask = sampling_mask
    params.TVWeight = 0.0002
    params.wavWeight = 0.0006
    iterations = 8
    params.data = kSpace
    
    recon = np.zeros_like(fftkSpaceShifted, np.complex)
    
    
    
    
    
    start = time.time() #time generation
    zf_image = cs_utils.ifft2u(kSpace)
    #zf_image = cs_utils.ifft2u(kSpace/sampling_pdf)
    
    wavelet_x0 = cs_utils.dw2(zf_image)
    wavelet_x0_coeff = wavelet_x0.coeffs
    wavelet_x0_coeffabs = np.abs(wavelet_x0_coeff)
    
    #compute reconstruction
    wavelet_x = cs_utils.dw2(zf_image)
    params.data = kSpace
    for k in range(1, iterations):
        wavelet_x = fnlCg(wavelet_x, params)
        recon = cs_utils.idw2(wavelet_x)
    # recon = recon * 255 / np.max(np.abs(recon))\
    recon /= N
    
    # image = image * 255 / np.max(np.abs(image))
    
    zf_image = zf_image * 255 / np.max(np.abs(zf_image))
    
    print("Done")
    end = time.time()
    end = time.time()
    elapsed[j] = end - start
    print("CS Reconstruction Took " + str(elapsed[j]) + " secs or " + str(elapsed[j]/60) + " mins")  
    rmse[j] = np.sqrt(metrics.mean_squared_error(np.abs(image*255/np.max(np.abs(image))), np.abs(recon*255/np.max(np.abs(recon)))))
    ssim[j] = metrics.structural_similarity(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    psnr[j] = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    print("RMSE:", rmse)
    print("SSIM:", ssim)
    print("PSNR:", psnr)
    
    diff = np.abs(image) - np.abs(recon)
    
    plt.figure(1)
    plt.imshow(np.abs(sampling_mask))
    
    fig = plt.figure(2)
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(np.abs(recon), cmap='gray')
    
    fig = plt.figure(3)
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(np.abs(image), cmap='gray')
    
    fig = plt.figure(4)
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(np.abs(zf_image), cmap='gray')
    
    plt.figure(9)
    plt.imshow(np.abs(diff))
    
    recon /= np.max(np.abs(recon))
    recon *= 255
    if fractal:
        im = Image.fromarray(np.rot90(np.abs(zf_image), k=3).astype(np.uint8))
        im.save("knee_lustig/fractal/" + "zf_" + "R" + str(R) + '.png')
        im = Image.fromarray(np.rot90(np.abs(recon), k=3).astype(np.uint8))
        im.save("knee_lustig/fractal/" + "R" + str(R) + '.png')
        im = Image.fromarray(np.rot90(np.abs(sampling_mask*255), k=3).astype(np.uint8))
        im.save("knee_lustig/fractal/" + "mask_" + "R" + str(R) + '.png')
        
    elif cart2d:
        im = Image.fromarray(np.rot90(np.abs(zf_image), k=3).astype(np.uint8))
        im.save("knee_lustig/2d/" + "zf_" + "R" + str(R) + '.png')
        im = Image.fromarray(np.rot90(np.abs(recon), k=3).astype(np.uint8))
        im.save("knee_lustig/2d/" + "R" + str(R) + '.png')
        im = Image.fromarray(np.abs(sampling_mask*255).astype(np.uint8))
        im.save("knee_lustig/2d/" + "mask_" + "R" + str(R) + '.png')
        
    else:
        im = Image.fromarray(np.rot90(np.abs(zf_image), k=3).astype(np.uint8))
        im.save("knee_lustig/1d/" + "zf_" + "R" + str(R) + '.png')
        im = Image.fromarray(np.rot90(np.abs(recon), k=3).astype(np.uint8))
        im.save("knee_lustig/1d/" + "R" + str(R) + '.png')
        im = Image.fromarray(np.rot90(np.abs(sampling_mask*255), k=3).astype(np.uint8))
        im.save("knee_lustig/1d/" + "mask_" + "R" + str(R) + '.png')

if fractal:
    savemat("knee_lustig/fractal/lustig_knee_frac" + ".mat", {'time':elapsed, 'rmse':rmse, 'psnr':psnr, 'ssim':ssim, 'r':r})
elif cart2d:
    savemat("knee_lustig/2d/lustig_knee_2d" + ".mat", {'time':elapsed, 'rmse':rmse, 'psnr':psnr, 'ssim':ssim, 'r':r})
else:
    savemat("knee_lustig/1d/lustig_knee_1d" + ".mat", {'time':elapsed, 'rmse':rmse, 'psnr':psnr, 'ssim':ssim, 'r':r})