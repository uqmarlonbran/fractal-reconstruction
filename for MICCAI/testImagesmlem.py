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
import finite
from fareyFractal import farey_fractal
from scipy import ndimage 
import matplotlib.pyplot as plt
import scipy.io as io
from makeRandomFractal import makeRandomFractal
import imageio as imageio #local module

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

#parameters
floatType = np.complex
#-------------------------------
#create test image
images = np.array(imageio.imread('data/mri/paper_samples/sub-OAS30002_ses-d0653_run-01_T1w_00105.png'), dtype=np.complex)
images = images * 255 / np.max(np.abs(images))
kspace = fftpack.fft2(images)

[N1, N2] = images.shape
N = N1
#-------------------------------


#-------------------------------
#compute lines
centered = False
circle = False
R = 4
tilingSize=6
if R == 2:
    fareyOrder = 10
    K = 2.4
elif R == 3:
    fareyOrder = 8
    K = 1.3
elif R == 4:
    fareyOrder = 7
    K = 0.88
    tilingSize=10
elif R == 8:
    fareyOrder = 5
    K = 0.3
    tilingSize=11

# Generate the fractal
# lines, angles, mValues, fractalMine, _, oversampleFilter = farey_fractal(N, fareyOrder, centered=centered)
# Setup fractal
if circle:
    lines, angles, \
        mValues, fractalMine, \
        oversampleFilter = finite.finiteFractal(N, K, sortBy='Euclidean', 
                                            twoQuads=True)
    lines = np.array(lines)
else:
    lines, angles, mValues, fractalMine, _, oversampleFilter = farey_fractal(N, fareyOrder, centered=centered)

p = N

if N%2==0:
    projs = int(N+N/2)
else:
    projs = N+1

drtSpace = np.zeros((projs, p), np.complex)
for i, line in enumerate(lines):
    u, v = line
    sliceReal = ndimage.map_coordinates(np.real(kspace), [u,v])
    sliceImag = ndimage.map_coordinates(np.imag(kspace), [u,v])
    slice = sliceReal+1j*sliceImag
    finiteProjection = (fftpack.ifft(slice)) # recover projection using slice theorem
    drtSpace[mValues[i], :] = finiteProjection

BL = -255 * (1 + 1j)
BU = 255 * (1 + 1j)


# Sample kspace
undersampleF = kspace * fractalMine

print("Samples used: ", R)

it = 251

smoothType = 3

lmd = [0.004, 1.0e5, 5]

# Reconstruct each of the brain images
start = time.time()
#3 is best for shrew and 2 for the lego
recon, firstRecon, psnrArr, ssimArr = iterative.abmlem_frt_complex(it, N, drtSpace, mValues, BU, BL, 3, 5, oversampleFilter, h=8, metric=True, ground=np.abs(images))
# recon = (recon) / np.max(np.abs(recon))
# recon *= 255

# images = np.round(images)
# recon = np.round(recon)

end = time.time()
elapsed = end - start
print("FFTSIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")  
mse = metrics.mean_squared_error(np.abs(images), np.abs(recon))
ssim = metrics.structural_similarity(np.abs(images).astype(float), np.abs(recon).astype(float), data_range=255)
psnr = metrics.peak_signal_noise_ratio(np.abs(images), np.abs(recon), data_range=255)
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

diff = np.abs(images - recon)

index = 0

plt.figure(1)
plt.imshow(np.abs(fractalMine))

fig = plt.figure(2)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(recon), cmap='gray', vmin=0, vmax=255)
# fig.savefig("output_fftsirt/test_number_" + str(index)+"reduction_factor_" + str(R) + '.png', bbox_inches='tight', pad_inches=0)


fig = plt.figure(3)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(images), cmap='gray', vmin=0, vmax=255)

plt.figure(4)
plt.imshow(np.abs(firstRecon))
plt.figure(9)
plt.imshow(np.abs(diff))


plt.figure(10)
plt.plot(range(0, it), psnrArr)
plt.figure(11)
plt.plot(range(0, it), ssimArr)



# # Plot prediction results
# fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20,5))
# ax[0].axis('off')
# ax[1].axis('off')
# ax[2].axis('off')
# ax[3].axis('off')
# ax[0].imshow(np.abs(images), aspect="auto", cmap='gray', vmin=0, vmax=255)
# ax[1].imshow(np.abs(fftpack.fftshift(fractalMine)), aspect="auto", cmap='gray', vmin=0, vmax=1)
# ax[2].imshow(np.abs(firstRecon), aspect="auto", cmap='gray', vmin=0, vmax=255)
# ax[3].imshow(np.abs(recon), aspect="auto", cmap='gray', vmin=0, vmax=255)
# ax[0].set_title("Ground Truth", fontsize=15)
# ax[1].set_title("Sampling Pattern", fontsize=15)
# ax[2].set_title("Zero Fill", fontsize=15)
# ax[3].set_title("Prediction", fontsize=15)
# fig.tight_layout()
# output_path = "output_fftsirt/test_number_"+str(index)+'.png'
# fig.savefig(output_path)
# plt.close(fig)