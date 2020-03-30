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
from makeRandomFractal import makeRandomFractal
import finitetransform.samplers as samplers
from PIL import Image
from scipy.io import loadmat, savemat
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
images = fftpack.fftshift(fftpack.ifft2(dftSpace))[:, :, :]
dftSpace = np.zeros_like(images, dtype=np.complex)
imMax = np.max(np.abs(images))
#Attention: You must ensure the kspace data is correctly centered or not centered.
for i, im in enumerate(images):
    dftSpace[i, :, :] = fftpack.fft2(fftpack.ifftshift(im)) * 255 / imMax
    dftSpace[i, :, :] = np.roll(dftSpace[i, :, :], 1, axis=0)

images = fftpack.ifft2(dftSpace)

#Attention: You must ensure the kspace data is correctly centered or not centered.

dftSpace = fftpack.fft2(images)
image = fftpack.ifft2(dftSpace[0,:,:])
dftSpace = dftSpace[0,:,:]

im = Image.fromarray(np.abs(fftpack.fftshift(image)*255/np.max(np.abs(image))).astype(np.uint8))
im.save("knee_fftsirt/knee" + '.png')

#compute lines
centered = False
#compute lines
fractal =True
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
        a=2
    elif R == 4:
        fareyOrder = 7
        K = 0.88
        tilingSize=8
        a=3.3
    elif R == 8:
        fareyOrder = 5
        K = 0.3
        tilingSize=8
        a=10
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
    
    tilingSize=32
    if cart2d:
        sampler, params = create_sampler('cart_2d', M=N, r=R, acs=tilingSize, N=N)
        sampling_mask = fftpack.ifftshift(sampler.generate_mask((N,N)))
    elif not fractal:
        sampler, params = create_sampler('cart_1d', M=N, r=R, acs=tilingSize, N=N)
        sampling_mask = fftpack.ifftshift(sampler.generate_mask((N,N)))
        
    fractalMine = sampling_mask
    sampling_mask = fftpack.fftshift(sampling_mask)
    undersampleF = np.zeros_like(dftSpace, np.complex)
    undersampleF = dftSpace * fractalMine
    a = np.sum(fractalMine) / N**2
    print("Samples used: ", a)
    
    t = (N**2) # reduction factor 0.5
    it = 200
    
    recon = np.zeros_like(images, np.complex)
    firstRecon = np.zeros_like(dftSpace, np.complex)
    
    h = [4, 6, 6, 6]
    
    lmd = [0.001, 1.0e5, 1.05]
    smoothType = 3
    # Reconstruct each of the brain images
    start = time.time()
    # USE NON-LOCAL WITH A h=8
    recon, firstRecon, psnrArr, ssimArr = iterative.sirt_fft_complex(it, N, undersampleF, fractalMine, t, smoothType, 3, h=h[j], lmd=lmd, metric=True, k=True, ground=np.abs(image), insertSamples=True)
    recon, _ = iterative.sirt_fft_complex(6, N, fftpack.fft2(recon), fractalMine, t, 1, 3, h=1, lmd=lmd, metric=False, k=True, ground=np.abs(image), insertSamples=True)
    recon = fftpack.fftshift(recon)
    firstRecon = fftpack.fftshift(firstRecon)
    imageComp = fftpack.fftshift(image)
    
    end = time.time()
    elapsed[j] = end - start
    print("FFT SIRT Reconstruction took " + str(elapsed[j]) + " secs or " + str(elapsed[j]/60) + " mins")
    rmse[j] = np.sqrt(metrics.mean_squared_error(np.abs(imageComp*255/np.max(np.abs(imageComp))), np.abs(recon*255/np.max(np.abs(recon)))))
    ssim[j] = metrics.structural_similarity(np.abs(imageComp), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    psnr[j] = metrics.peak_signal_noise_ratio(np.abs(imageComp), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    print("RMSE:", rmse)
    print("SSIM:", ssim)
    print("PSNR:", psnr)
    
    
    diff = np.abs(image - recon)
    
    index = 0
    
    plt.figure(1)
    plt.imshow(np.abs(fractalMine))
    
    fig = plt.figure(2)
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(np.abs(recon), cmap='gray')
    # fig.savefig("output_fftsirt/test_number_" + str(index)+"reduction_factor_" + str(R) + '.png', bbox_inches='tight', pad_inches=0)
    
    
    fig = plt.figure(3)
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(np.abs(image), cmap='gray')
    
    plt.figure(4)
    plt.imshow(np.abs(firstRecon))
    plt.figure(9)
    plt.imshow(np.abs(diff), cmap='gray')
    
    
    plt.figure(10)
    plt.plot(range(0, it), psnrArr)
    plt.figure(11)
    plt.plot(range(0, it), ssimArr)

    recon /= np.max(np.abs(recon))
    recon *= 255
    zf_image = np.abs(firstRecon) * 255 / np.max(np.abs(firstRecon))
    if fractal:
        im = Image.fromarray(np.rot90(np.abs(zf_image), k=3).astype(np.uint8))
        im.save("knee_fftsirt/fractal/" + "zf_" + "R" + str(R) + '.png')
        im = Image.fromarray(np.rot90(np.abs(recon), k=3).astype(np.uint8))
        im.save("knee_fftsirt/fractal/" + "R" + str(R) + '.png')
        im = Image.fromarray(np.rot90(np.abs(sampling_mask*255), k=3).astype(np.uint8))
        im.save("knee_fftsirt/fractal/" + "mask_" + "R" + str(R) + '.png')
        
    elif cart2d:
        im = Image.fromarray(np.abs(zf_image).astype(np.uint8))
        im.save("knee_fftsirt/2d/" + "zf_" + "R" + str(R) + '.png')
        im = Image.fromarray(np.abs(recon).astype(np.uint8))
        im.save("knee_fftsirt/2d/" + "R" + str(R) + '.png')
        im = Image.fromarray(np.abs(sampling_mask*255).astype(np.uint8))
        im.save("knee_fftsirt/2d/" + "mask_" + "R" + str(R) + '.png')
        
    else:
        im = Image.fromarray(np.abs(zf_image).astype(np.uint8))
        im.save("knee_fftsirt/1d/" + "zf_" + "R" + str(R) + '.png')
        im = Image.fromarray(np.abs(recon).astype(np.uint8))
        im.save("knee_fftsirt/1d/" + "R" + str(R) + '.png')
        im = Image.fromarray(np.abs(sampling_mask*255).astype(np.uint8))
        im.save("knee_fftsirt/1d/" + "mask_" + "R" + str(R) + '.png')

if fractal:
    savemat("knee_fftsirt/fractal/fftsirt_knee_frac" + ".mat", {'time':elapsed, 'rmse':rmse, 'psnr':psnr, 'ssim':ssim, 'r':r})
elif cart2d:
    savemat("knee_fftsirt/2d/fftsirt_knee_2d" + ".mat", {'time':elapsed, 'rmse':rmse, 'psnr':psnr, 'ssim':ssim, 'r':r})
else:
    savemat("knee_fftsirt/1d/fftsirt_knee_1d" + ".mat", {'time':elapsed, 'rmse':rmse, 'psnr':psnr, 'ssim':ssim, 'r':r})