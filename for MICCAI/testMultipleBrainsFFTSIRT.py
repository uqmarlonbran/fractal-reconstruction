# -*- coding: utf-8 -*-
"""
Load Brain images and test FFT SIRT reconstruction from square fractal sampling

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
import finite
import iterativeReconstruction as iterative
from makeRandomFractal import makeRandomFractal
# from fareyFractal import farey_fractal
# from scipy import ndimage 
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

def round_sig(x, sig=3):
    return np.round(x, sig-int(np.floor(np.log10(np.abs(x))))-1)

#parameters
N = 256
floatType = np.complex
#-------------------------------
#load kspace data
images, num_cases = loadNii.load_data("slices_pad", singlePres=True)
images = images[1414:, :, :]
dftSpace = np.zeros_like(images, np.complex)
#load Cartesian data
#Attention: You must ensure the kspace data is correctly centered or not centered.
#-------------------------------

#compute the Cartesian reconstruction for comparison
print("Computing Chaotic Reconstruction...")

for i, im in enumerate(images):
    dftSpace[i, :, :] = fftpack.fft2(im)

#-------------------------------
#compute lines
centered = False
random = True
R = 4
tilingSize=8
if R == 2:
    fareyOrder = 10
    K = 2.4
elif R == 3:
    fareyOrder = 8
    K = 1.3
elif R == 4:
    fareyOrder = 7
    K = 0.88
    tilingSize=9
elif R == 8:
    fareyOrder = 5
    K = 0.3
    tilingSize=11

# Generate the fractal
# lines, angles, mValues, fractalMine, _, oversampleFilter = farey_fractal(N, fareyOrder, centered=centered)
# Setup fractal
if not random:
    lines, angles, \
        mValues, fractalMine, \
        oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', 
                                            twoQuads=True)
    lines = np.array(lines)
else:
    fractalMine, samplingFilter = makeRandomFractal(N, (1/R) * 0.9, tilingSize=tilingSize, withTiling=True)
# Sample kspace
undersampleF = np.zeros_like(dftSpace, np.complex)
for d, dft in enumerate(dftSpace):
    undersampleF[d, :, :] = dft * fractalMine

print("Samples used: ", R)

t = (N**2)*1.5 # reduction factor 0.5
it = 250
#t = 50 # reduction factor 0.18
#it = 1250

[N1, N2, N3] = np.shape(images)

recon = np.zeros_like(images, np.complex)
firstRecon = np.zeros_like(images, np.complex)
diff = np.zeros_like(images, np.complex)

mse = np.zeros((N1, 1))
psnr = np.zeros((N1, 1))
ssim = np.zeros((N1, 1))
elapsed = np.zeros((N1, 1))

smoothType = 3

lmd = [0.001, 1.0e5, 5]

# Reconstruct each of the brain images
for k, kspace in enumerate(undersampleF):
    start = time.time()
    recon[k,:,:], firstRecon[k,:,:] = iterative.sirt_fft_complex(it, N, kspace, fractalMine, t, smoothType, 3, h=8, lmd=lmd, insertSamples=False)
    end = time.time()
    elapsed[k,:] = end - start
    mse[k,:] = metrics.mean_squared_error(np.abs(images[k, :, :]), np.abs(recon[k, :, :]))
    ssim[k,:] = metrics.structural_similarity(np.abs(images[k, :, :]), np.abs(recon[k, :, :]), data_range=255)
    psnr[k,:] = metrics.peak_signal_noise_ratio(np.abs(images[k, :, :]), np.abs(recon[k, :, :]), data_range=255)
    diff[k,:,:] = np.abs(images[k,:,:] - recon[k,:,:])
    print(k)
    
print("Average Time: ", round_sig(np.average(elapsed)))
print("Average MSE: ", round_sig(np.average(mse)))
print("Average PSNR: ", round_sig(np.average(psnr)))
print("Average SSIM: ", round_sig(np.average(ssim)))

savemat("FFTSIRT_" + str(R) + "_BRAINS.mat", {'time':elapsed, 'mse':mse, 'psnr':psnr, 'ssim':ssim})

