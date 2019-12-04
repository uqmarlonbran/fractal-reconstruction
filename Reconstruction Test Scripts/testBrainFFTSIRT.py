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
import finitetransform.imageio as imageio #local module
import loadNii
import numpy as np
import time
import math
import iterativeReconstruction as iterative
from fareyFractal import farey_fractal

import matplotlib.pyplot as plt

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

#parameters
N = 256
floatType = np.complex
twoQuads = True
p = N


#-------------------------------
#load kspace data
images, num_cases = loadNii.load_data("slices_pad")
dftSpace = np.zeros_like(images, np.complex)
#load Cartesian data
#Attention: You must ensure the kspace data is correctly centered or not centered.

#-------------------------------
maxValues = []
minValues = []
#compute the Cartesian reconstruction for comparison
print("Computing Chaotic Reconstruction...")
for k, image in enumerate(images):
    if k == 1:
        break
    print("Loading image: ", k)
    images[k, :, :] /= np.max(image)
    images[k, :, :] *= 255
    dftSpace[k, :, :] = fftpack.fft2(image)
    maxValues.append(np.max(np.abs(image)))
    minValues.append(np.min(np.abs(image)))

print("Images Max Value:", np.max(maxValues))
print("Images Min Value:", np.min(minValues))


#-------------------------------
#compute lines
centered = False

# Generate the fractal
lines, angles, mValues, fractalMine, R, oversampleFilter = farey_fractal(N, 11, centered=centered)
lines = np.array(lines)
# Sample kspace
undersampleF = np.zeros_like(dftSpace, np.complex)
undersampleF = dftSpace * fractalMine

print("Samples used: ", R)

t = (N**2)*1.5 # reduction factor 0.5
it = 2500
#t = 50 # reduction factor 0.18
#it = 1250

recon = np.zeros_like(images, np.complex)
firstRecon = np.zeros_like(images, np.complex)

# Reconstruct each of the brain images
start = time.time()
for i, F in enumerate(undersampleF):
    recon[i, :, :], firstRecon[i, :, :] = iterative.sirt_fft_complex(it, N, F, fractalMine, t, 3, 50)
    print("Image number: ", i)
    if i >= 0:
        break
    
end = time.time()
elapsed = end - start
print("FFT SIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")
recon = np.abs(recon)
images = np.abs(images)
diff = np.abs(image - recon)

mse = imageio.immse(np.abs(images[0, :, :]), np.abs(recon[0, :, :]))
ssim = imageio.imssim(np.abs(images[0, :, :]).astype(float), np.abs(recon[0, :, :]).astype(float))
psnr = imageio.impsnr(np.abs(images[0, :, :]), np.abs(recon[0, :, :]))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

plt.figure(2)
plt.imshow(np.abs(recon[0, :, :]))
plt.figure(3)
plt.imshow(np.abs(images[0, :, :]))
plt.figure(4)
plt.imshow(np.abs(firstRecon[0, :, :]))
plt.figure(9)
plt.imshow(np.abs(diff[0, :, :]))
plt.figure(10)
plt.imshow(np.abs(fftpack.ifft2(dftSpace[0, :, :] * fractalMine)))