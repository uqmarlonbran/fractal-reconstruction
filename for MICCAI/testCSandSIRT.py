# -*- coding: utf-8 -*-
"""
Compute the CS reconstruction via the NLCG, Lustig's algorithm

Created on Fri Nov  9 09:26:46 2018

@author: shakes
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './fareyFunctions')
# from __future__ import print_function    # (at top of module)
import _libpath #add custom libs
import numpy as np
import time

#cs
import param_class
import cs_utils_original as cs_utils
import scipy.fftpack as fftpack
import pyfftw
from fnlCg_original import fnlCg
import scipy.io as io
import iterativeReconstruction as iterative
from scipy import ndimage
import finitetransform.imageio as imageio #local module
import math
# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

#parameters
N = 512
k = 1
M = k*N
K = 0.1
r = 0.48
iterations = 8
twoQuads = True
print("N:", N, "M:", M)

brain_dict = io.loadmat('brain512.mat')
kspace = brain_dict.get('data')
fftkSpace = kspace/(np.abs(kspace)).max()
sampling_mask = brain_dict.get('mask')
sampling_pdf = brain_dict.get('pdf')

#mask FFT space
sampling_mask = sampling_mask.astype(np.uint32)
kSpace = fftkSpace*sampling_mask
#print("kSpace:", kSpace)
print("kSpace Shape:", kSpace.shape)

#Define parameter class for nlcg
params = param_class.nlcg_param()
params.FTMask = sampling_mask
params.TVWeight = 0.0002
params.wavWeight = 0.0005
params.data = kSpace

start = time.time() #time generation
#zf_image = cs_utils.ifft2u(kSpace)
zf_image = cs_utils.ifft2u(kSpace/sampling_pdf)

wavelet_x0 = cs_utils.dw2(zf_image)
wavelet_x0_coeff = wavelet_x0.coeffs
wavelet_x0_coeffabs = np.abs(wavelet_x0_coeff)

#compute reconstruction
wavelet_x = wavelet_x0
for i in range(1, iterations):
    wavelet_x = fnlCg(wavelet_x, params)
    recon = cs_utils.idw2(wavelet_x)
recon = np.abs(recon)
    
print("Done")
end = time.time()
elapsed = end - start
print("CS Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    
#-------------------------------
#plot slices responsible for reconstruction
import matplotlib.pyplot as plt
plt.figure(11)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))

fontsize = 18
plt.rc('xtick', labelsize=fontsize-4) 
plt.rc('ytick', labelsize=fontsize-4) 

plt.gray()

ax[0].imshow(np.log10(np.abs(kSpace)))
ax[0].set_title("Fractal Sampling of K-space")
ax[1].imshow(np.abs(zf_image))
ax[1].set_title("Initial Reconstruction")
ax[2].imshow(recon)
ax[2].set_title("Fractal CS Reconstruction")

plt.tight_layout()

#wavelets
plt.figure()
plt.imshow(np.abs(wavelet_x0_coeffabs))
plt.title("DWT")

plt.show()


######### SIRT Reconstruction

t = (N**2)*1.5 # reduction factor 0.5
it = 301

reconSIRT = np.zeros_like(recon, np.complex)
firstRecon = np.zeros_like(recon, np.complex)

smoothType = 1

# kspace = fftpack.ifftshift(kspace)
sampling_mask = fftpack.ifftshift(sampling_mask)
images = fftpack.ifft2(kspace)
images = fftpack.fftshift(images * 255 / np.max(np.abs(images)))
kSpaceSirt = fftpack.fft2(images)
kSpaceSirt = fftpack.ifftshift(kSpaceSirt)
images = fftpack.ifft2(kSpaceSirt)
# Reconstruct each of the brain images
start = time.time()
reconSIRT, firstRecon = iterative.sirt_fft_complex(it, 512, kSpaceSirt*sampling_mask, sampling_mask, t, smoothType, 5, h=16, lmd=0.0005)
reconSIRT = (reconSIRT * 255) / np.max(np.abs(reconSIRT))


end = time.time()
elapsed = end - start
print("FFTSIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")  


print("CS + TV Recon Vals:")
recon = recon * 255 / np.max(np.abs(recon))
mse = imageio.immse(np.abs(images), np.abs(recon))
ssim = imageio.imssim(np.abs(images/np.abs(np.max(images))).astype(float), np.abs(reconSIRT/np.abs(np.max(reconSIRT))).astype(float))
psnr = imageio.impsnr(np.abs(images), np.abs(recon))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

print("SIRT Recon Vals:")
mse = imageio.immse(np.abs(images), np.abs(reconSIRT))
ssim = imageio.imssim(np.abs(images/np.abs(np.max(images))).astype(float), np.abs(reconSIRT/np.abs(np.max(reconSIRT))).astype(float))
psnr = imageio.impsnr(np.abs(images), np.abs(reconSIRT))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

index = 2
diff = np.abs(images - reconSIRT)
plt.figure(1)
plt.imshow(np.abs(sampling_mask))

fig = plt.figure(2)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(reconSIRT), cmap='gray', vmin=0, vmax=255)
# fig.savefig("output_fftsirt/test_number_" + str(index)+"reduction_factor_" + str(R) + '.png', bbox_inches='tight', pad_inches=0)

plt.figure(3)
plt.imshow(np.abs(images))
plt.figure(4)
plt.imshow(np.abs(firstRecon))
plt.figure(9)
plt.imshow(np.abs(diff))


# Plot prediction results
# fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20,5))
# ax[0].axis('off')
# ax[1].axis('off')
# ax[2].axis('off')
# ax[3].axis('off')
# ax[0].imshow(np.abs(images), aspect="auto", cmap='gray', vmin=0, vmax=255)
# ax[1].imshow(np.abs(fftpack.fftshift(sampleMask)), aspect="auto", cmap='gray', vmin=0, vmax=1)
# ax[2].imshow(np.abs(firstRecon), aspect="auto", cmap='gray', vmin=0, vmax=255)
# ax[3].imshow(np.abs(recon), aspect="auto", cmap='gray', vmin=0, vmax=255)
# ax[0].set_title("Ground Truth")
# ax[1].set_title("Sampling Pattern")
# ax[2].set_title("Zero Fill")
# ax[3].set_title("Prediction")
# fig.tight_layout()
# output_path = "output_fftsirt/test_number_"+str(index)+'.png'
# fig.savefig(output_path)
# plt.close(fig)