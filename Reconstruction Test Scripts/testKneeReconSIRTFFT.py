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
N = 320
floatType = np.complex
twoQuads = True
p = N

#-------------------------------
# Load fractals
print("Loading the fractals...")
fractals, num_fractals = loadNii.load_data("knees/fractal")
# Fractal indexes
shakes = 1
marlon = 0
tiled = 2
tiled2 = 3

#-------------------------------
#load images data
images, num_cases = loadNii.load_data("knees/knee_slices_reduced")

# load Cartesian data
dftSpace, num_cases = loadNii.load_data_channels("knees/knee_kspace_slices_reduced")
#Attention: You must ensure the kspace data is correctly centered or not centered.


#-------------------------------
#compute lines
centered = False

# Generate the fractal
# lines, angles, mValues, fractalMine, R, oversampleFilter = farey_fractal(N, 13, centered=centered)
# lines = np.array(lines)
fractalMine = fftpack.ifftshift(fractals[tiled, :, :])
# Sample kspace
# newDFT = np.zeros_like(dftSpace[0, :, :, :])
# newIm = np.zeros_like(newDFT)
undersampleF = np.zeros_like(dftSpace, np.complex)
for k, dft in enumerate(dftSpace):
    undersampleF[k, :, :, :] = dft * fractalMine
R = np.sum(fractalMine) / N**2
print("Samples used: ", R)

t = (N**2) # reduction factor 0.5
it = 1025

recon = np.zeros_like(images, np.complex)
firstRecon = np.zeros_like(dftSpace, np.complex)

# Reconstruct each of the brain images
start = time.time()
# USE NON-LOCAL WITH A h=8
for i, F in enumerate(undersampleF):
    if i == 100:
        recon[i, :, :], firstRecon[i, :, :, :] = iterative.sirt_fft_complex_multi(it, N, F, fractalMine, t, 3, 25, complexOutput=True, h=12) # 50
        recon[i, :, :] = fftpack.fftshift(recon[i, :, :])
        print("Image number: ", i)
    
newFirstRecon = np.zeros_like(recon, dtype=np.complex)
for i, ims in enumerate(firstRecon):
    for im in ims:
        newFirstRecon[i, :, :] += (im ** 2)

    newFirstRecon[i, :, :] = np.abs(np.sqrt(newFirstRecon[i, :, :]))
    
end = time.time()
elapsed = end - start
for i, image in enumerate(images):
    images[i, :, :] = image * 255 / np.max(np.abs(image))
    images[i, :, :] = fftpack.fftshift(image)
print("FFT SIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")
recon = np.abs(recon)
images = np.abs(images)
diff = np.abs(images - recon)
mse = imageio.immse(images[100,:,:], np.abs(recon[100, :, :]))
ssim = imageio.imssim((images[100,:,:]).astype(float), np.abs(recon[100, :, :]).astype(float))
psnr = imageio.impsnr(np.abs(images[100,:,:]), np.abs(recon[100, :, :]))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

plt.figure(2)
plt.imshow(np.abs(recon[100, :, :]))
plt.figure(3)
plt.imshow(np.abs(images[100,:,:]))
plt.figure(9)
plt.imshow(np.abs(diff[100, :, :]))
plt.figure(10)
plt.imshow(np.abs(fftpack.ifftshift(newFirstRecon[100, :, :])))