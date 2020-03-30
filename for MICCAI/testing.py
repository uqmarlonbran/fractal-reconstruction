# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:22:51 2020

@author: s4358744
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './fareyFunctions')
# from __future__ import print_function    # (at top of module)
import _libpath #add custom libs
import numpy as np
import time
import matplotlib.pyplot as plt

import scipy.fftpack as fftpack
import pyfftw
import scipy.io as io
import iterativeReconstruction as iterative

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

#parameters
N = 512

brain_dict = io.loadmat('brain512.mat')
kspace = fftpack.ifftshift(brain_dict.get('data'))
sampling_mask = fftpack.ifftshift(brain_dict.get('mask'))
sampling_pdf = brain_dict.get('pdf')

brain_image = fftpack.ifftshift(fftpack.ifft2(kspace*sampling_mask))
brain_image = brain_image * 255 / np.max(np.abs(brain_image))
kspace = fftpack.fft2(brain_image)

t = (N**2)*1.5 # reduction factor 0.5
it = 251
#t = 50 # reduction factor 0.18
#it = 1250

smoothType = 1

lmd = [0.002, 1.0e10, 5]

# Reconstruct each of the brain images
start = time.time()
recon, firstRecon = iterative.sirt_fft_complex(it, N, kspace, sampling_mask, t, smoothType, 5, h=15, lmd=lmd)
recon = (recon * 255) / np.max(np.abs(recon))

end = time.time()
elapsed = end - start
print("FFTSIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")  
# mse = imageio.immse(np.abs(images), np.abs(recon))
# ssim = imageio.imssim(np.abs(images/np.max(np.abs(images))).astype(float), np.abs(recon/np.max(np.abs(recon))).astype(float))
# psnr = imageio.impsnr(np.abs(images), np.abs(recon))
# print("RMSE:", math.sqrt(mse))
# print("SSIM:", ssim)
# print("PSNR:", psnr)

# diff = np.abs(images - recon//1)

index = 0


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))

fontsize = 18
plt.rc('xtick', labelsize=fontsize-4) 
plt.rc('ytick', labelsize=fontsize-4) 

plt.gray()

ax[0].imshow(np.log10(np.abs(fftpack.fftshift(kspace))))
ax[0].set_title("Fractal Sampling of K-space")
ax[1].imshow(np.abs(firstRecon))
ax[1].set_title("Initial Reconstruction")
ax[2].imshow(np.abs(recon))
ax[2].set_title("Fractal CS Reconstruction")

plt.tight_layout()

plt.show()

# plt.figure(1)
# plt.imshow(np.abs(sampling_mask))

# fig = plt.figure(2)
# plt.axis('off')
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.imshow(np.abs(recon), cmap='gray', vmin=0, vmax=255)
# # fig.savefig("output_fftsirt/test_number_" + str(index)+"reduction_factor_" + str(R) + '.png', bbox_inches='tight', pad_inches=0)

# fig = plt.figure(4)
# plt.axis('off')
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.imshow(np.abs(firstRecon), cmap='gray', vmin=0, vmax=255)



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