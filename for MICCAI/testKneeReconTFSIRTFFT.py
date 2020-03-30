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
lines, angles, mValues, fractalMine, R, oversampleFilter = farey_fractal(N, 15, centered=centered)
lines = np.array(lines)
# fractalMine = fftpack.ifftshift(fractals[marlon, :, :])
# Sample kspace
# newDFT = np.zeros_like(dftSpace[0, :, :, :])
# newIm = np.zeros_like(newDFT)
undersampleF = np.zeros_like(dftSpace, np.complex)
for k, dft in enumerate(dftSpace):
    undersampleF[k, :, :, :] = dft * fractalMine
R = np.sum(fractalMine) / N**2
print("Samples used: ", R)

t = (N**2) # reduction factor 0.5
it = 251

# recon = np.zeros_like(images, np.complex)
smoothType = 3
# Reconstruct each of the brain images
start = time.time()

# USE NON-LOCAL WITH A h=8
for i, F in enumerate(undersampleF):
    if i == 0:
        c1, c2, c3, c4, c5, c6, c7, c8 = iterative.tf_sirt_fft_complex_multi(it, N, F, fractalMine, t, smoothType, 5, complexOutput=True, h=10) # 50
        c1 = fftpack.fftshift(c1)
        c2 = fftpack.fftshift(c2)
        c3 = fftpack.fftshift(c3)
        c4 = fftpack.fftshift(c4)
        c5 = fftpack.fftshift(c5)
        c6 = fftpack.fftshift(c6)
        c7 = fftpack.fftshift(c7)
        c8 = fftpack.fftshift(c8)
        print("Image number: ", i)

newImage = np.zeros((N, N), dtype=np.complex)
# Combine the data from each channel

newImage += (c1 ** 2)
newImage += (c2 ** 2)
newImage += (c3 ** 2)
newImage += (c4 ** 2)
newImage += (c5 ** 2)
newImage += (c6 ** 2)
newImage += (c7 ** 2)
newImage += (c8 ** 2)


newImage = np.sqrt(newImage)
# if not complexOutput:
#     newImage = np.absolute(newImage)
newImage = newImage * 255 / np.max(np.abs(newImage))
recon = newImage
    
end = time.time()
elapsed = end - start
for i, image in enumerate(images):
    images[i, :, :] = image * 255 / np.max(np.abs(image))
    images[i, :, :] = fftpack.fftshift(image)
print("FFT SIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")

diff = np.abs(images - recon)
recon = np.abs(recon)
images = np.abs(images)
mse = imageio.immse(images[0,:,:], np.abs(recon))
ssim = imageio.imssim(np.abs(images[0, :, :]/np.abs(np.max(images[0,:,:]))).astype(float), np.abs(recon/np.abs(np.max(recon))).astype(float))
psnr = imageio.impsnr(np.abs(images[0,:,:]), np.abs(recon))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

plt.figure(1)
plt.imshow(np.abs(fractalMine))
plt.figure(2)
plt.imshow(np.abs(recon))
plt.figure(3)
plt.imshow(np.abs(images[0,:,:]))
plt.figure(9)
plt.imshow(np.abs(diff[0, :, :]))
# plt.figure(10)
# plt.imshow(np.abs(fftpack.ifftshift(newFirstRecon[0, :, :])))