# -*- coding: utf-8 -*-
"""
Test of ABfMLEM using Shepp Logan phantom and squared fractal sampling

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
#bounds
BL = -100*(1 + 1j)
BU = 100*(1 + 1j)

#load Cartesian data
#Attention: You must ensure the kspace data is correctly centered or not centered.
image = shepp_logan(N)
kspace = (fftpack.fft2(image))
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
#samples used
sampleNumber = lines.shape[0] * N
print("Samples used:", sampleNumber, ", proportion:", sampleNumber/float(N*N))
print("Lines proportion:", lines.shape[0]/float(N))
#-------------
# Measure finite slice
from scipy import ndimage
print("Measuring slices")
powSpectGrid = np.abs(dftSpace)
if N%2==0:
    projs = int(N+N/2)
else:
    projs = N+1
drtSpace = np.zeros((projs, p), floatType)
for i, line in enumerate(lines):
    u, v = line
    sliceReal = ndimage.map_coordinates(np.real(dftSpace), [u,v])
    sliceImag = ndimage.map_coordinates(np.imag(dftSpace), [u,v])
    slice = sliceReal+1j*sliceImag
    finiteProjection = (fftpack.ifft(slice)) # recover projection using slice theorem
    drtSpace[mValues[i],:] = finiteProjection


recon = np.abs(finite.ifrt_complex(drtSpace, N, mValues=mValues))
testdrtSpace = finite.frt_complex(image, N)



start = time.time()
recon, firstRecon = iterative.abmlem_frt_complex(300, N, drtSpace, mValues, BU, BL, 2, 1, oversampleFilter=oversampleFilter)
end = time.time()
elapsed = end - start
print("ABMLEM Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")
testdrtSpace = finite.frt_complex(image, N)

recon = np.abs(recon)

sample = dftSpace * fractalMine

mse = imageio.immse(image, np.abs(recon))
ssim = imageio.imssim(image.astype(float), recon.astype(float))
psnr = imageio.impsnr(image, np.abs(recon))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)
diff = image - recon

plt.figure(1)
plt.imshow(np.abs(drtSpace))
plt.figure(2)
plt.imshow(np.abs(recon))
plt.figure(3)
plt.imshow(np.abs(image))
plt.figure(4)
plt.imshow(np.abs(firstRecon))
plt.figure(5)
plt.imshow(np.abs(finite.frt_complex(recon, N)))
plt.figure(6)
plt.imshow(fractalMine)
plt.figure(7)
plt.imshow(np.abs(testdrtSpace))
plt.figure(8)
plt.imshow(np.abs(finite.ifrt_complex(testdrtSpace, N, mValues=mValues, oversampleFilter=oversampleFilter)))
plt.figure(9)
plt.imshow(np.abs(diff))
plt.figure(10)
plt.imshow(np.abs(fftpack.ifft2(sample)))