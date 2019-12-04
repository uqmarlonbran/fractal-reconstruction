# -*- coding: utf-8 -*-
"""
Load Brain images and test ABfMLEM reconstruction from square fractal sampling

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
from scipy import ndimage
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

if N%2==0:
    projs = int(N+N/2)
else:
    projs = N+1
#-------------------------------
#load kspace data
images, num_cases = loadNii.load_data("slices_pad")
dftSpace = np.zeros_like(images, np.complex)
drtSpace = np.zeros((num_cases, projs, p), np.complex)
#load Cartesian data
#Attention: You must ensure the kspace data is correctly centered or not centered.
#-------------------------------
#compute lines
centered = False
lines, angles, mValues, fractalMine, R, oversampleFilter = farey_fractal(N, 11, centered=centered)
lines = np.array(lines)

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
    
    for i, line in enumerate(lines):
        u, v = line
        sliceReal = ndimage.map_coordinates(np.real(dftSpace[k, :, :]), [u,v])
        sliceImag = ndimage.map_coordinates(np.imag(dftSpace[k, :, :]), [u,v])
        slice = sliceReal+1j*sliceImag
        finiteProjection = (fftpack.ifft(slice)) # recover projection using slice theorem
        drtSpace[k, mValues[i], :] = finiteProjection

print("Images Max Value:", np.max(maxValues))
print("Images Min Value:", np.min(minValues))

scale = 1.3
#BL = (np.min(minValues) - 10) * (1 + 1j) * scale
#BU = (np.max(maxValues) + 10) * (1 + 1j) * scale
BL = -200 * (1 + 1j)
BU = 400 * (1 + 1j)


# Sample kspace
undersampleF = np.zeros_like(dftSpace, np.complex)
undersampleF = dftSpace * fractalMine

print("Samples used: ", R)

it = 251

recon = np.zeros_like(images, np.complex)
firstRecon = np.zeros_like(images, np.complex)

# Reconstruct each of the brain images
start = time.time()
for i, g_j in enumerate(drtSpace):
    recon[i, :, :], firstRecon[i, :, :] = iterative.abmlem_frt_complex(it, N, g_j, mValues, BU, BL, 3, 10, oversampleFilter)
    print("Image number: ", i)
    if i >= 0:
        break
    
end = time.time()
elapsed = end - start
print("ABfMLEM Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")
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