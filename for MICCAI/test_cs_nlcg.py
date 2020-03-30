# -*- coding: utf-8 -*-
"""
Compute the CS reconstruction via the NLCG, Lustig's algorithm

Created on Fri Nov  9 09:26:46 2018

@author: shakes
"""
# from __future__ import print_function    # (at top of module)
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './fareyFunctions')
import _libpath #add custom libs
import finitetransform.farey as farey #local module
import imageio as imageio #local module
import finitetransform.imageio as imageoi
import finitetransform.radon as radon
import numpy as np
import finite
import time
import math

#cs
import param_class
import cs_utils
import scipy.fftpack as fftpack
import pyfftw
from fnlCg import fnlCg
#import scipy.io as io

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

#parameters
N = 256
k = 1
M = k*N
K = 0.1
r = 0.48
iterations = 16
twoQuads = True
print("N:", N, "M:", M)

fareyVectors = farey.Farey()        
fareyVectors.compactOn()
fareyVectors.generateFiniteWithCoverage(N)

#sort to reorder result for prettier printing
finiteAnglesSorted, anglesSorted = fareyVectors.sort('length')
print("Finite Coverage mu:", len(finiteAnglesSorted))

#create test image
# image, mask = imageio.lena(N, M, True, np.float, True) #non-prime
# image, mask = imageio.phantom(N, M, True, np.float, True) #non-prime
# image, mask = imageio.cameraman(N, M, True, np.float, True) #non-prime
image = imageio.imread('data/phantom.png')

#-------------------------------
#k-space
# Fourier Space of Image
import scipy.fftpack as fftpack

#2D FFT
fftSpace = fftpack.fft2(image) #the '2' is important
fftkSpaceShifted = fftpack.fftshift(fftSpace)
print("FFT Shape:", fftkSpaceShifted.shape)
#power spectrum
powSpect = np.abs(fftkSpaceShifted)
fftkSpaceShifted = fftkSpaceShifted/(powSpect).max()
ifftkSpaceShifted = fftpack.fftshift(fftkSpaceShifted)
imageNorm = fftpack.ifft2(ifftkSpaceShifted) #the '2' is important
imageNorm = np.real(imageNorm)
powSpect = np.zeros((N,N))
#-------------------------------
#compute finite lines
centered = False
lines, angles, mValues = finite.computeRandomLines(powSpect, anglesSorted, finiteAnglesSorted, r, K, centered, True)
mu = len(lines)
print("Number of finite lines:", len(lines))
print("Number of finite points:", len(lines)*(M-1))

#sampling mask
sampling_mask = np.zeros((M,M), np.float)
for line in lines:
    u, v = line
    for x, y in zip(u, v):
        sampling_mask[x, y] += 1
#determine oversampling because of power of two size
#this is fixed for choice of M and m values
oversamplingFilter = np.zeros((M,M), np.float)
onesSlice = np.ones(M, np.float)
for m in mValues:
    radon.setSlice(m, oversamplingFilter, onesSlice, 2)
oversamplingFilter[oversamplingFilter==0] = 1
sampling_mask /= oversamplingFilter
sampling_mask = fftpack.fftshift(sampling_mask)

#tile center region further
radius = N/8
centerX = M/2
centerY = M/2
count = 0
for i, row in enumerate(sampling_mask):
    for j, col in enumerate(row):
        distance = math.sqrt( (i-float(centerX))**2 + (j-float(centerY))**2)
        if distance < radius:
            if not sampling_mask[i, j] > 0: #already selected
                count += 1
                sampling_mask[i, j] = 1
totalSamples = mu*(M-1)+count+1
actualR = float(totalSamples/M**2)
print("Number of total sampled points:", totalSamples)
print("Actual Reduction factor:", actualR)
imageio.imsave("sampling_fractal.png", sampling_mask)

#mask FFT space
sampling_mask = sampling_mask.astype(np.uint32)
kSpace = fftkSpaceShifted*sampling_mask
#print("kSpace:", kSpace)
print("kSpace Shape:", kSpace.shape)

#Define parameter class for nlcg
params = param_class.nlcg_param()
params.FTMask = sampling_mask
params.TVWeight = 0.0003 
params.wavWeight = 0.0005
params.data = kSpace

start = time.time() #time generation
zf_image = cs_utils.ifft2u(kSpace)
#zf_image = cs_utils.ifft2u(kSpace/sampling_pdf)

wavelet_x0 = cs_utils.dw2(zf_image)
wavelet_x0_coeff = wavelet_x0.coeffs
wavelet_x0_coeffabs = np.abs(wavelet_x0_coeff)

#compute reconstruction
wavelet_x = wavelet_x0
for i in range(1, iterations):
    wavelet_x = fnlCg(wavelet_x, params)
    recon = cs_utils.idw2(wavelet_x)
recon = np.abs(recon / np.max(np.abs(recon)))
imageNorm = np.abs(imageNorm / np.max(np.abs(imageNorm)))
    
print("Done")
end = time.time()
elapsed = end - start
print("CS Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    
mse = imageoi.immse(imageNorm, recon)
ssim = imageoi.imssim(imageNorm.astype(float), recon.astype(float))
psnr = imageoi.impsnr(imageNorm, recon)
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)
diff = image - recon

#save mat file of result
#np.savez('result_cs.npz', recon=recon, diff=diff)
np.savez('result_phantom_cs.npz', recon=recon, diff=diff)
#np.savez('result_camera_cs.npz', recon=recon, diff=diff)
    
#-------------------------------
#plot slices responsible for reconstruction
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))

fontsize = 18
plt.rc('xtick', labelsize=fontsize-4) 
plt.rc('ytick', labelsize=fontsize-4) 

plt.gray()

ax[0].imshow(imageNorm)
ax[0].set_title("Original Image")
ax[1].imshow(np.log10(np.abs(kSpace)))
ax[1].set_title("Fractal Sampling of K-space")
ax[2].imshow(np.real(zf_image))
ax[2].set_title("Initial Reconstruction")
ax[3].imshow(recon)
ax[3].set_title("Fractal CS Reconstruction")
#ax[4].imshow(diff)
#ax[4].set_title("Fractal-based Back-projection")

plt.tight_layout()

#wavelets
plt.figure()
plt.imshow(np.abs(wavelet_x0_coeffabs))
plt.title("DWT")

#kspace contour
fig, ax = plt.subplots()

x = np.linspace(-1.0, 1.0, N)
y = np.linspace(-1.0, 1.0, N)
X, Y = np.meshgrid(x, y)

cs = ax.contourf(X, Y, powSpect, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
ax.set_title("k-Space")
cbar = fig.colorbar(cs)

plt.show()
