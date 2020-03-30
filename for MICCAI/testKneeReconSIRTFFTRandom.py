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

#-------------------------------
#load images data
# images, num_cases = loadNii.load_data("knees/knee_slices_reduced")

# load Cartesian data
dftSpace, num_cases = loadNii.load_data_channels("knees/knee_kspace_slices_reduced")
dftSpace = np.squeeze(dftSpace)
images = fftpack.ifft2(dftSpace)
imMax = np.max(np.abs(images))
#Attention: You must ensure the kspace data is correctly centered or not centered.
for i, im in enumerate(images):
    dftSpace[i, :, :] = fftpack.fft2(im) * 255 / imMax
    dftSpace[i, :, :] = np.roll(dftSpace[i, :, :], 1, axis=0)

images = fftpack.ifft2(dftSpace)
#compute lines
centered = False
random = True
cartesian = False
R = 2
tilingSize=8
if R == 2:
    fareyOrder = 10
    K = 2.4
elif R == 3:
    fareyOrder = 8
    K = 1.3
    tilingSize=8
elif R == 4:
    fareyOrder = 7
    K = 0.88
    tilingSize=10
elif R == 8:
    fareyOrder = 5
    K = 0.3
    tilingSize=11
elif R == 10:
    tilingSize=13

# Generate the fractal
# lines, angles, mValues, fractalMine, _, oversampleFilter = farey_fractal(N, fareyOrder, centered=centered)
# Setup fractal
if not random:
    lines, angles, \
        mValues, sampling_mask, \
        oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', 
                                            twoQuads=True)
    lines = np.array(lines)
else:
    sampling_mask, samplingFilter = makeRandomFractal(N, (1/R), tilingSize=tilingSize, withTiling=True)
    
if cartesian:
    sampling_mask, r_factor= iterative.generate_mask_alpha(size=[N,N], r_factor_designed=R, r_alpha=2,seed=-1)
fractalMine = sampling_mask
undersampleF = np.zeros_like(dftSpace, np.complex)
undersampleF = dftSpace * fractalMine
R = np.sum(fractalMine) / N**2
print("Samples used: ", R)

t = (N**2) * 0.75 # reduction factor 0.5
it = 200

recon = np.zeros_like(images, np.complex)
firstRecon = np.zeros_like(dftSpace, np.complex)

lmd = [0.001, 1.0e5, 1.05]
smoothType = 3
# Reconstruct each of the brain images
start = time.time()
# USE NON-LOCAL WITH A h=8
recon, firstRecon = iterative.sirt_fft_complex_multi(it, N, undersampleF, fractalMine, t, 3, 5, complexOutput=True, h=8, lmd=lmd) # 50
recon = fftpack.fftshift(recon)
firstRecon = fftpack.fftshift(iterative.rss(firstRecon))
image = fftpack.fftshift(iterative.rss(images))

end = time.time()
elapsed = end - start
print("FFT SIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")
mse = metrics.mean_squared_error(np.abs(image), np.abs(recon))
ssim = metrics.structural_similarity(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
psnr = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
print("RMSE:", math.sqrt(mse))
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


# plt.figure(10)
# plt.plot(range(0, it), psnrArr)
# plt.figure(11)
# plt.plot(range(0, it), ssimArr)