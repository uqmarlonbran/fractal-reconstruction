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
from fnlCg import fnlCg
import param_class
import cs_utils
from scipy.io import loadmat, savemat
# from fareyFractal import farey_fractal
# from scipy import ndimage 
import matplotlib.pyplot as plt
import scipy.io as io
import finitetransform.samplers as samplers
from makeRandomFractal import makeRandomFractal

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
#load kspace data
mouse=False
if mouse:
    brain_dict = io.loadmat('data/shrew_Cartesian.mat')
else:   
    brain_dict = io.loadmat('data/Cartesian_LEGO.mat')
    
dftSpace = brain_dict.get('Cartesian_data')
powSpect = np.abs(dftSpace)
fftkSpaceShifted = dftSpace/(np.abs(powSpect)).max()
image = fftpack.fftshift(fftpack.ifft2(fftkSpaceShifted))
fftkSpaceShifted = fftpack.fft2(image)

def create_sampler(mode, M=256, r=0.5, r_alpha=2, acs=0., seed=-1, 
                       N=256, k=1, K=0.1, s=8, twoQuads=True):
    if 'cart' in mode.lower():
        params = {'M': M, 'r': r, 'r_alpha': r_alpha, 'acs': acs}
        if '1d' in mode.lower():
            sampler = samplers.OneDimCartesianRandomSampler(r=r, r_alpha=r_alpha, acs=acs, seed=-1)
        elif '2d' in mode.lower():
            sampler = samplers.TwoDimCartesianRandomSampler(r=r, r_alpha=r_alpha, acs=acs, seed=-1)
    elif 'fractal' in mode.lower():
        params = {'N': N, 'k': k, 'M': M, 'K': K, 'r': r, 's': s,
                'ctr': acs, 'twoQuads': twoQuads}
        sampler = samplers.FractalSampler(k, K, r, acs, s, seed=-1)
    return sampler, params

[N, N1] = image.shape
#-------------------------------
#-------------------------------
#compute lines
centered = False
random = True
cartesian = True
#compute lines
centered = False
#compute lines
fractal=False
random=True
cart2d=False
R = 8
tilingSize=8
if cartesian == False:
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
        tilingSize=8
        R = 3.3
    elif R == 8:
        fareyOrder = 5
        K = 0.3
        tilingSize=8
        R = 10
    elif R == 6:
        tilingSize=8
        R = 6

# Generate the fractal
# lines, angles, mValues, fractalMine, _, oversampleFilter = farey_fractal(N, fareyOrder, centered=centered)
# Setup fractal
if fractal:
    if not random:
        lines, angles, \
            mValues, sampling_mask, \
            oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', 
                                                twoQuads=True)
        lines = np.array(lines)
    else:
        sampling_mask, samplingFilter = makeRandomFractal(N, (1/R) * 0.9, tilingSize=tilingSize, withTiling=True)
    


#-------------------------------

tilingSize=32
if cart2d:
    sampler, params = create_sampler('cart_2d', M=N, r=R, acs=tilingSize, N=N)
    sampling_mask = fftpack.ifftshift(sampler.generate_mask((N,N)))
elif not fractal:
    sampler, params = create_sampler('cart_1d', M=N, r=R, acs=tilingSize, N=N)
    sampling_mask = fftpack.ifftshift(sampler.generate_mask((N,N)))
    
sampling_mask = fftpack.fftshift(sampling_mask)
# print("Samples used: ", R)
kSpace = fftkSpaceShifted*sampling_mask    

#Define parameter class for nlcg
params = param_class.nlcg_param()
params.FTMask = sampling_mask
params.TVWeight = 0.0004
params.wavWeight = 0.0004
iterations = 8
params.data = kSpace

recon = np.zeros_like(fftkSpaceShifted, np.complex)





start = time.time() #time generation
zf_image = cs_utils.ifft2u(kSpace)
#zf_image = cs_utils.ifft2u(kSpace/sampling_pdf)

wavelet_x0 = cs_utils.dw2(zf_image)
wavelet_x0_coeff = wavelet_x0.coeffs
wavelet_x0_coeffabs = np.abs(wavelet_x0_coeff)

#compute reconstruction
wavelet_x = cs_utils.dw2(zf_image)
params.data = kSpace
for k in range(1, iterations):
    wavelet_x = fnlCg(wavelet_x, params)
    recon = cs_utils.idw2(wavelet_x)
# recon = recon * 255 / np.max(np.abs(recon))
# recon /= N

imagea = image* N
# image = fftpack.fftshift(image)
# recon = fftpack.fftshift(recon)
# zf_image = fftpack.fftshift(zf_image)
zf_image = zf_image * 255 / np.max(np.abs(zf_image))

print("Done")
end = time.time()
end = time.time()
elapsed = end - start
print("CS Reconstruction Took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")  
mse = metrics.mean_squared_error(np.abs(imagea)*255/np.max(np.abs(imagea)), np.abs(recon)*255/np.max(np.abs(recon)))
ssim = metrics.structural_similarity(np.abs(imagea), np.abs(recon), data_range=0.2)
psnr = metrics.peak_signal_noise_ratio(np.abs(imagea), np.abs(recon), data_range=0.2)
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

diff = np.abs(imagea) - np.abs(recon)

plt.figure(1)
plt.imshow(np.abs(sampling_mask))

fig = plt.figure(2)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(recon), cmap='gray')

fig = plt.figure(3)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(imagea), cmap='gray')

fig = plt.figure(4)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(zf_image), cmap='gray')

plt.figure(9)
plt.imshow(np.abs(diff))

# if cartesian:
#     savemat("LUSTIG_Cartesian_LEGO.mat", {'time':elapsed, 'mse':mse, 'psnr':psnr, 'ssim':ssim, 'r',:r})
# else:
    
