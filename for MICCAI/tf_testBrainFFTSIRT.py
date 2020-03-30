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
import finitetransform.imageio as imageio #local module
import loadNii
import numpy as np
import time
import math
import finite
import iterativeReconstruction as iterative
# from fareyFractal import farey_fractal
# from scipy import ndimage 
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
#-------------------------------
#load kspace data
images, num_cases = loadNii.load_data("slices_pad", singlePres=True, num=1200)
dftSpace = np.zeros_like(images, np.complex)
#load Cartesian data
#Attention: You must ensure the kspace data is correctly centered or not centered.
#-------------------------------
#-------------------------------
maxValues = []
minValues = []
#compute the Cartesian reconstruction for comparison
print("Computing Chaotic Reconstruction...")

dftSpace = fftpack.fft2(images)
maxValues.append(np.max(np.abs(images)))
minValues.append(np.min(np.abs(images)))

print("Images Max Value:", np.max(maxValues))
print("Images Min Value:", np.min(minValues))


#-------------------------------
#compute lines
centered = False
R = 2
if R == 2:
    fareyOrder = 10
    K = 2.4
elif R == 3:
    fareyOrder = 8
    K = 1.3
elif R == 4:
    fareyOrder = 7
    K = 0.88
elif R == 8:
    fareyOrder = 5
    K = 0.3

# Generate the fractal
# lines, angles, mValues, fractalMine, _, oversampleFilter = farey_fractal(N, fareyOrder, centered=centered)
# Setup fractal
lines, angles, \
    mValues, fractalMine, \
    oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', 
                                        twoQuads=True)
lines = np.array(lines)
# Sample kspace
undersampleF = np.zeros_like(dftSpace, np.complex)
undersampleF = dftSpace * fractalMine

print("Samples used: ", R)

t = (N**2)*1.5 # reduction factor 0.5
it = 251
#t = 50 # reduction factor 0.18
#it = 1250

recon = np.zeros_like(images, np.complex)
firstRecon = np.zeros_like(images, np.complex)

smoothType = 3

# Reconstruct each of the brain images
start = time.time()
recon, firstRecon = iterative.tf_sirt_fft_complex(it, N, undersampleF, fractalMine, t, smoothType, 5, h=6, lmd=0.001)
recon = (recon * 255) / np.max(np.abs(recon))

end = time.time()
elapsed = end - start
print("FFTSIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")  
mse = imageio.immse(np.abs(images), np.abs(recon))
ssim = imageio.imssim(np.abs(images/np.max(np.abs(images))).astype(float), np.abs(recon/np.max(np.abs(recon))).astype(float))
psnr = imageio.impsnr(np.abs(images), np.abs(recon))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

diff = np.abs(images - recon/1)

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

plt.figure(3)
plt.imshow(np.abs(images))
plt.figure(4)
plt.imshow(np.abs(firstRecon))
plt.figure(9)
plt.imshow(np.abs(diff))



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