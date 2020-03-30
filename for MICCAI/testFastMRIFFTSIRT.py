# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 10:57:30 2020

@author: marlo
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './fareyFunctions')
import scipy.fftpack as fftpack
import pyfftw
import skimage.metrics as metrics
import numpy as np
import time
import math
import finite
import iterativeReconstruction as iterative
import h5py
import matplotlib.pyplot as plt
import scipy.io as io
from makeRandomFractal import makeRandomFractal

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()


def show_slices(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        


#parameters
file = 'FastMRI/file_brain_AXT2_200_2000158.h5'
hf = h5py.File(file)
print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))

# Get kspace data from h5 file
volume_kspace = hf['kspace'][()]
mask = hf['mask'][()]
print(volume_kspace.dtype)
print(volume_kspace.shape)

[slices, coils, N1, N2] = volume_kspace.shape

# Multi-Coil: (number of slices, number of coils, height, width)


# Chose the ith slice of this volume
# kspace = volume_kspace[10, :, N1//2 - N2//2:N1//2 + N2//2, :]
kspace = volume_kspace[5, :, :, :]
images = np.zeros_like(kspace)

# Centre the image
for i, k in enumerate(kspace):
    k = fftpack.ifftshift(k)
    # kspace = np.roll(kspace, 1, axis=0) #fix 1 pixel shift
    images[i, :, :] = fftpack.fftshift(fftpack.ifft2(k))
    images[i, :, :] = images[i, :, :] * 255 / np.max(np.abs(images[i, :, :]))
    # kspace[i, :, :] = fftpack.fft2(images[i, :, :])

# images = complex_center_crop(images, (coils, N2, N2))


# Crop the image
images = images[:, N1//2 - N2//2:N1//2 + N2//2, :]
# images = images[:, 1:318, 1:318]
kspace = np.zeros_like(images)
# mask = mask[1:318]

# Get sampling pattern
[coils, N1, N2] = kspace.shape
fractalMine = fftpack.ifftshift(np.tile(mask, (N1, 1)))
N = fractalMine.shape

# Get k-space
for i, im in enumerate(images):
    kspace[i, :, :] = fftpack.fft2(im) * fractalMine
    images[i, :, :] = fftpack.ifft2(kspace[i, :, :])




show_slices(np.abs(images), range(0, coils), cmap='gray')

plt.figure(3)
plt.imshow(np.abs(iterative.rss(images)), cmap='gray')

# fftkSpace = kspace/(np.abs(kspace)).max()
# sampling_mask = brain_dict.get('mask')
# sampling_pdf = brain_dict.get('pdf')
#load Cartesian data
#Attention: You must ensure the kspace data is correctly centered or not centered.
#-------------------------------


#-------------------------------
# #compute lines
# centered = False
# random = False
# R = 2
# tilingSize=8
# if R == 2:
#     fareyOrder = 10
#     K = 2.4
# elif R == 3:
#     fareyOrder = 8
#     K = 1.3
# elif R == 4:
#     fareyOrder = 7
#     K = 0.88
#     tilingSize=10
# elif R == 8:
#     fareyOrder = 5
#     K = 0.3
#     tilingSize=12

# # Generate the fractal
# # lines, angles, mValues, fractalMine, _, oversampleFilter = farey_fractal(N, fareyOrder, centered=centered)
# # Setup fractal
# if not random:
#     lines, angles, \
#         mValues, fractalMine, \
#         oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', 
#                                             twoQuads=True)
#     lines = np.array(lines)
# else:
#     fractalMine, samplingFilter = makeRandomFractal(N, (1/R) * 0.9, tilingSize=tilingSize, withTiling=True)

# Sample kspace
# undersampleF = np.zeros_like(kspace, np.complex)
# undersampleF = kspace * fractalMine

# print("Samples used: ", R)

t = (320**2)*1.5 # reduction factor 0.5
it = 100
#t = 50 # reduction factor 0.18
#it = 1250

recon = np.zeros_like(images, np.complex)
firstRecon = np.zeros_like(images, np.complex)

smoothType = 3

lmd = [0.001, 1.0e5, 5]

# Reconstruct each of the brain images
start = time.time()
recon, firstRecon = iterative.sirt_fft_complex_multi(it, N, kspace, fractalMine, t, smoothType, 2, h=12, lmd=lmd)
recon = (recon * 255) / np.max(np.abs(recon))

end = time.time()
elapsed = end - start
print("FFTSIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")  
# mse = metrics.mean_squared_error(np.abs(images), np.abs(recon))
# ssim = metrics.structural_similarity(np.abs(images).astype(float), np.abs(recon).astype(float), data_range=255)
# psnr = metrics.peak_signal_noise_ratio(np.abs(images), np.abs(recon), data_range=255)
# print("RMSE:", math.sqrt(mse))
# print("SSIM:", ssim)
# print("PSNR:", psnr)

# diff = np.abs(images - recon)

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

# plt.figure(3)
# plt.imshow(np.abs(images))
# plt.figure(4)
# plt.imshow(np.abs(firstRecon))
# plt.figure(9)
# plt.imshow(np.abs(diff))



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