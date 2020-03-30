
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
import skimage.metrics as metrics
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
from makeRandomFractal import makeRandomFractal

import finitetransform.samplers as samplers
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

N = 320
#-------------------------------
# load Cartesian data
dftSpace, num_cases = loadNii.load_data_channels("knees/knee_kspace_slices_reduced")
dftSpace = np.squeeze(dftSpace)/np.abs(np.max(dftSpace))
images = fftpack.ifft2(dftSpace)
imMax = np.max(np.abs(images))
#Attention: You must ensure the kspace data is correctly centered or not centered.
for i, im in enumerate(images):
    dftSpace[i, :, :] = fftpack.fft2(fftpack.fftshift(im))

fftkSpaceShifted = fftpack.fftshift(dftSpace)
images = fftpack.ifft2(dftSpace)

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

#-------------------------------
#compute lines
fractal =False
cart2d=True
R = 8
tilingSize=32
if fractal:
    sampler, params = create_sampler('cart_fractal', M=N, r=R, acs=tilingSize, N=N)
    sampling_mask = sampler.generate_mask((N,N))
elif cart2d:
    sampler, params = create_sampler('cart_2d', M=N, r=R, acs=tilingSize, N=N)
    sampling_mask = sampler.generate_mask((N,N))
else:
    sampler, params = create_sampler('cart_1d', M=N, r=R, acs=tilingSize, N=N)
    sampling_mask = sampler.generate_mask((N,N))
# print("Samples used: ", R)
kSpace = np.zeros_like(fftkSpaceShifted, dtype=np.complex)
for i, k in enumerate(fftkSpaceShifted):
    kSpace[i,:,:] = k*sampling_mask    

#Define parameter class for nlcg
params = param_class.nlcg_param()
params.FTMask = sampling_mask
params.TVWeight = 0.0004
params.wavWeight = 0.0004
iterations = 2
params.data = kSpace[0,:,:]

recons = np.zeros_like(fftkSpaceShifted, np.complex)



start = time.time() #time generation

zf_image = np.zeros_like(kSpace)
for i, dft in enumerate(kSpace):

    zf_image[i,:,:] = cs_utils.ifft2u(kSpace[i, :, :])
#zf_image = cs_utils.ifft2u(kSpace/sampling_pdf)

wavelet_x0 = cs_utils.dw2(zf_image[0,:,:])
wavelet_x0_coeff = wavelet_x0.coeffs
wavelet_x0_coeffabs = np.abs(wavelet_x0_coeff)

#compute reconstruction
for i, im in enumerate(zf_image):
    wavelet_x = cs_utils.dw2(im)
    params.data = kSpace[i,:,:]
    for k in range(1, iterations):
        wavelet_x = fnlCg(wavelet_x, params)
        recons[i,:,:] = cs_utils.idw2(wavelet_x)
    
    # recon[i,:,:] = (recon[i,:,:] * 255) / np.max(recon[i,:,:]) 
    
recon = iterative.rss(recons / N)
firstRecon = iterative.rss(zf_image)
# firstRecon = firstRecon * 255 / np.max(np.abs(firstRecon))

image = iterative.rss(images) 
# images = images * 255 / np.max(np.abs(images))
print("Done")
end = time.time()
end = time.time()
elapsed = end - start
print("CS Reconstruction Took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")  
mse = metrics.mean_squared_error(np.abs(image), np.abs(recon))
ssim = metrics.structural_similarity(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
psnr = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
print("RMSE:", math.sqrt(mse))
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

diff = np.abs(image) - np.abs(recon)

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
plt.imshow(np.abs(image), cmap='gray')

fig = plt.figure(4)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(firstRecon), cmap='gray')

plt.figure(9)
plt.imshow(np.abs(diff))