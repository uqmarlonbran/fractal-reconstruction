# -*- coding: utf-8 -*-
"""
Test of FFT SIRT using Shepp Logan phantom and squared fractal sampling

@author: marlon
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './fareyFunctions')
from phantominator import shepp_logan
import finitetransform.numbertheory as nt #local modules
import scipy.fftpack as fftpack
import pyfftw
import finitetransform.imageio as imageio #local module

import numpy as np
import finite as finite
import time
import math
import iterativeReconstruction as iterative
from fareyFractal import farey_fractal, normalizer

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
p = nt.nearestPrime(N)
p = N


#-------------------------------
#load kspace data
from scipy.io import loadmat

#load Cartesian data
#Attention: You must ensure the kspace data is correctly centered or not centered.
image = shepp_logan(N)
kspace = fftpack.fft2(image)
print("kSpace Shape:", kspace.shape)
kMaxValue = np.max(kspace)
kMinValue = np.min(kspace)
print("k-Space Max Value:", kMaxValue)
print("k-Space Min Value:", kMinValue)
print("k-Space Max Magnitude:", np.abs(kMaxValue))
print("k-Space Min Magnitude:", np.abs(kMinValue))

#-------------------------------
#compute the Cartesian reconstruction for comparison
print("Computing Chaotic Reconstruction...")
dftSpace = kspace
image = fftpack.ifft2(dftSpace) #the '2' is important
image = np.abs(image)
maxValue = np.max(image)
minValue = np.min(image)

print("Image Max Value:", maxValue)
print("Image Min Value:", minValue)


#-------------------------------
#compute lines
centered = False

# Generate the fractal
lines, angles, mValues, fractalMine, R, oversampleFilter = farey_fractal(N, 10, centered=centered)
lines = np.array(lines)
# Sample kspace
F = dftSpace * fractalMine

#samples used
sampleNumber = lines.shape[0] * N
print("Samples used:", sampleNumber, ", proportion:", sampleNumber/float(N*N))
print("Lines proportion:", lines.shape[0]/float(N))
#-------------
# Measure finite slice
from scipy import ndimage
print("Measuring slices")
powSpectGrid = np.abs(dftSpace)

t = (N**2) # reduction factor 0.5
it = 300
#t = 50 # reduction factor 0.18
#it = 1250

start = time.time()
recon, firstRecon = iterative.sirt_fft_complex(it, N, F, fractalMine, t, 2, 1, centered=False)
end = time.time()
elapsed = end - start
print("FFT SIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")
testdrtSpace = finite.frt_complex(image, N)
image = np.abs(image)
recon = np.abs(recon)

diff = image - recon

mse = imageio.immse(image, np.abs(recon))
ssim = imageio.imssim(image.astype(float), recon.astype(float))
psnr = imageio.impsnr(image, np.abs(recon))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)
diff = image - recon

plt.figure(2)
plt.imshow(np.abs(recon))
plt.figure(3)
plt.imshow(np.abs(image))
plt.figure(4)
plt.imshow(np.abs(firstRecon))
plt.figure(9)
plt.imshow(np.abs(diff))