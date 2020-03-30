# -*- coding: utf-8 -*-
"""
Compute the CS reconstruction via the NLCG, Lustig's algorithm

Created on Fri Nov  9 09:26:46 2018

@author: shakes 


Modified on Mon Feb 10

@author: marlon
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './fareyFunctions')
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
import finitetransform.radon as radon
import scipy.fftpack as fftpack
import finitetransform.imageio as imageio #local module
import loadNii
import numpy as np
import time
import math
import random
import iterativeReconstruction as iterative
from fareyFractal import farey_fractal
from scipy import ndimage
import matplotlib.pyplot as plt
import finitetransform.farey as farey #local module
# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft


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
#-------------------------------
#k-space
# Fourier Space of Image

#parameters
N = 256
#-------------------------------
#load kspace data
images, num_cases = loadNii.load_data("slices_pad", num=1200, singlePres=True)
dftSpace = np.zeros_like(images, np.complex)
#load Cartesian data
#Attention: You must ensure the kspace data is correctly centered or not centered.
# dftSpace = cs_utils.fft2u(images, params)
fftkSpaceShifted = fftpack.fftshift(fftpack.fft2(images))
powSpect = np.abs(fftkSpaceShifted)
fftkSpaceShifted = fftkSpaceShifted/(powSpect).max()
ifftkSpaceShifted = fftpack.fftshift(fftkSpaceShifted)



#-------------------------------
sampleMask = np.zeros((N, N), dtype=np.complex)
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
## Generate random sampling
# R = 2
# fullySample = 24
# # Fully sampling centre 32 lines
# randInts = np.array(range(fullySample//2, N-(fullySample//2)))
# random.shuffle(randInts)
# randInts = randInts[0:N//R - fullySample]

# sampleMask[randInts,:] = np.ones((1,N))

# sampleMask[0:fullySample//2,:] = np.ones((1,N))
# sampleMask[N-fullySample//2-1:, :] = np.ones((1,N))

# sampleMask = fftpack.fftshift(sampleMask).transpose()


# ## Compute CS
# sampling_mask = sampleMask



#Define parameter class for nlcg
params = param_class.nlcg_param()
params.FTMask = sampling_mask
params.TVWeight = 0.0004
params.wavWeight = 0.0004
params.data = dftSpace


kSpace = fftkSpaceShifted*sampling_mask
params.data = kSpace

recon = np.zeros_like(images, np.complex)
firstRecon = np.zeros_like(images, np.complex)



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
recon = (recon * 255) / np.max(recon) 
    
print("Done")
end = time.time()
end = time.time()
elapsed = end - start
print("CS Reconstruction Took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")  
mse = imageio.immse(np.abs(images), np.abs(recon))
ssim = imageio.imssim(np.abs(images/np.abs(np.max(images))).astype(float), np.abs(recon/np.abs(np.max(recon))).astype(float))
psnr = imageio.impsnr(np.abs(images), np.abs(recon))
print("SSIM:", ssim)
print("PSNR:", psnr)
# diff = imageNorm - recon

#save mat file of result
#np.savez('result_cs.npz', recon=recon, diff=diff)
# np.savez('result_phantom_cs.npz', recon=recon, diff=diff)
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

ax[0].imshow(np.abs(images))
ax[0].set_title("Original Image")
ax[1].imshow(np.abs(np.log10(kSpace)))
ax[1].set_title("Fractal Sampling of K-space")
ax[2].imshow(np.abs(zf_image))
ax[2].set_title("Initial Reconstruction")
ax[3].imshow(np.abs(recon))
ax[3].set_title("Fractal CS Reconstruction")
#ax[4].imshow(diff)
#ax[4].set_title("Fractal-based Back-projection")

plt.tight_layout()

#wavelets
plt.figure()
plt.imshow(np.abs(wavelet_x0_coeffabs))
plt.title("DWT")

# #kspace contour
# fig, ax = plt.subplots()

# x = np.linspace(-1.0, 1.0, N)
# y = np.linspace(-1.0, 1.0, N)
# X, Y = np.meshgrid(x, y)

# cs = ax.contourf(X, Y, powSpect, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
# ax.set_title("k-Space")
# cbar = fig.colorbar(cs)

plt.show()
