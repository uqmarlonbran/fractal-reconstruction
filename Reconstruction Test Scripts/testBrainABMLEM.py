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
lines, angles, mValues, fractalMine, R, oversampleFilter = farey_fractal(N, 10, centered=centered)
lines = np.array(lines)

#-------------------------------
maxValues = []
minValues = []
#compute the Cartesian reconstruction for comparison
print("Computing Chaotic Reconstruction...")


for k, image in enumerate(images):
    if k >= 1200 and k < 1203:
            
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
indexer = 1200
recon = np.zeros_like(images, np.complex)
firstRecon = np.zeros_like(images, np.complex)

# Reconstruct each of the brain images
start = time.time()
for i, g_j in enumerate(drtSpace):
    if i >= 1200 and i < 1201:
        recon[i, :, :], firstRecon[i, :, :] = iterative.abmlem_frt_complex(it, N, g_j, mValues, BU, BL, 3, 5, oversampleFilter, h=6)
        print("Image number: ", i)
    
end = time.time()
elapsed = end - start
print("ABfMLEM Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")
recon = np.abs(recon)
images = np.abs(images)
diff = np.abs(images - recon)

mse = imageio.immse(np.abs(images[indexer, :, :]), np.abs(recon[indexer, :, :]))
ssim = imageio.imssim(np.abs(images[indexer, :, :]).astype(float), np.abs(recon[indexer, :, :]).astype(float))
psnr = imageio.impsnr(np.abs(images[indexer, :, :]), np.abs(recon[indexer, :, :]))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

plt.figure(2)
plt.imshow(np.abs(recon[indexer, :, :]))
plt.figure(3)
plt.imshow(np.abs(images[indexer, :, :]))
plt.figure(4)
plt.imshow(np.abs(firstRecon[indexer, :, :]))
plt.figure(9)
plt.imshow(np.abs(diff[indexer, :, :]))
plt.figure(10)
plt.imshow(np.abs(fftpack.ifft2(dftSpace[indexer, :, :] * fractalMine)))

# # Plot prediction results
# fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
# for i in range(1200, 1203):
#     index = int(i)
#     fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
#     ax[0].imshow(np.abs(firstRecon[index,:,:]), aspect="auto")
#     ax[1].imshow(np.abs(images[index,:,:]), aspect="auto")
#     mask = np.abs(recon[index,:,:])
#     ax[2].imshow(mask, aspect="auto")
#     ax[0].set_title("Input")
#     ax[1].set_title("Ground truth")
#     ax[2].set_title("Prediction")
#     fig.tight_layout()
#     output_path = "output_abmlem/test_number_"+str(index)+'.png'
#     fig.savefig(output_path)
#     plt.close(fig)