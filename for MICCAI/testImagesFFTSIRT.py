# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:16:38 2020

@author: s4358744
"""

# from __future__ import print_function    # (at top of module)
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './fareyFunctions')
import _libpath #add custom libs
import finitetransform.farey as farey #local module
import imageio as imageio 
import skimage.metrics as metrics
import finitetransform.radon as radon
import numpy as np
import finite
import time
import math
import matplotlib.pyplot as plt
from PIL import Image


#cs
import param_class
import cs_utils
import scipy.fftpack as fftpack
import pyfftw
from fnlCg import fnlCg
import iterativeReconstruction as iterative
from makeRandomFractal import makeRandomFractal
#import scipy.io as io

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

#create test image
images = np.array(imageio.imread('data/mri/paper_samples/sub-OAS30002_ses-d0653_run-01_T1w_00105.png'), dtype=np.complex)
images = images * 255 / np.max(np.abs(images))
kspace = fftpack.fft2(images)

fractalsPath = "best_fractals/"

[N1, N2] = images.shape
N = N1

#compute lines
centered = False
random = True
bestRandom = True
R = 8
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
    tilingSize=8
elif R == 8:
    fareyOrder = 5
    K = 0.3
    tilingSize=10
elif R == 10:
    tilingSize=10

# Generate the fractal
# lines, angles, mValues, fractalMine, _, oversampleFilter = farey_fractal(N, fareyOrder, centered=centered)
# Setup fractal
if not random:
    lines, angles, \
        mValues, fractalMine, \
        oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', 
                                            twoQuads=True)
    lines = np.array(lines)
elif not bestRandom:
    fractalMine, samplingFilter = makeRandomFractal(N, (1/R) * 0.9, tilingSize=tilingSize, withTiling=True)
else:
    fractalMine = np.array(imageio.imread(fractalsPath + "r" + str(R) + "_best_mask.png"), dtype=np.complex)
    fractalMine = fftpack.ifftshift(fractalMine) // np.max(np.abs(fractalMine))
undersampleF = np.zeros_like(kspace, np.complex)
undersampleF = kspace * fractalMine

# print("Samples used: ", R)

t = (N**2)*1.5 # reduction factor 0.5
it = 100
#t = 50 # reduction factor 0.18
#it = 1250

smoothType = 3

lmd = [0.001, 1.0e5, 5]

# Reconstruct each of the brain images
start = time.time()
recon, firstRecon, psnrArr, ssimArr = iterative.sirt_fft_complex(it, N, undersampleF, fractalMine, t, smoothType, 3, h=4, lmd=lmd, metric=True, ground=np.abs(images), insertSamples=True)
# recon = (recon * 255) / np.max(np.abs(recon))

end = time.time()
elapsed = end - start
print("FFTSIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")  
mse = metrics.mean_squared_error(np.abs(images), np.abs(recon))
ssim = metrics.structural_similarity(np.abs(images).astype(float), np.abs(recon).astype(float), data_range=255)
psnr = metrics.peak_signal_noise_ratio(np.abs(images), np.abs(recon), data_range=255)
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

diff = np.abs(images - recon)

index = 0

plt.figure(1)
plt.imshow(np.abs(fractalMine))

fig = plt.figure(2)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(recon), cmap='gray', vmin=0, vmax=255)
# fig.savefig("output_fftsirt_miccai/reduction_factor_" + str(R) + '.png', bbox_inches='tight', pad_inches=0)
im = Image.fromarray(np.abs(recon).astype(np.uint8))
# im.save("output_fftsirt_miccai/reduction_factor_" + str(R) + '.png')


fig = plt.figure(3)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(images), cmap='gray', vmin=0, vmax=255)


fig = plt.figure(4)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(firstRecon), cmap='gray', vmin=0, vmax=255)
# fig.savefig("output_fftsirt_miccai/zf_reduction_factor_" + str(R) + '.png', bbox_inches='tight', pad_inches=0)
im = Image.fromarray(np.abs(firstRecon).astype(np.uint8))
# im.save("output_fftsirt_miccai/zf_reduction_factor_" + str(R) + '.png')

plt.figure(9)
plt.imshow(np.abs(diff))

plt.figure(10)
plt.plot(range(0, it), psnrArr)
plt.figure(11)
plt.plot(range(0, it), ssimArr)



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
# # fig.savefig(output_path)
# plt.close(fig)