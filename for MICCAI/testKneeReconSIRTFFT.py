# -*- coding: utf-8 -*-
"""
Load Knee images and test FFT SIRT reconstruction from the different sampling
schemes

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
import iterativeReconstruction as iterative
from fareyFractal import farey_fractal
import random
import matplotlib.pyplot as plt
from makeRandomFractal import makeRandomFractal
# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

#parameters
N = 320
floatType = np.complex
twoQuads = True
p = N

#-------------------------------
# Load fractals
print("Loading the fractals...")
fractals, num_fractals = loadNii.load_data("knees/fractal")
# Fractal indexes
shakes = 1
marlon = 0
tiled = 2
tiled2 = 3

#-------------------------------
#load images data
images, num_cases = loadNii.load_data("knees/knee_slices_reduced")

# load Cartesian data
dftSpace, num_cases = loadNii.load_data_channels("knees/knee_kspace_slices_reduced")
#Attention: You must ensure the kspace data is correctly centered or not centered.


#-------------------------------
#compute lines
centered = False

# Generate the fractal
# lines, angles, mValues, fractalMine, R, oversampleFilter = farey_fractal(N, 12, centered=centered)
# lines = np.array(lines)
# print("Reduction Factor: ", len(lines) / N)
fractalMine = fftpack.ifftshift(fractals[tiled2, :, :])

undersampleF = np.zeros_like(dftSpace, np.complex)
for k, dft in enumerate(dftSpace):
    undersampleF[k, :, :, :] = dft * fractalMine
R = np.sum(fractalMine) / N**2
print("Samples used: ", R)

t = (N**2) # reduction factor 0.5
it = 251

recon = np.zeros_like(images, np.complex)
firstRecon = np.zeros_like(dftSpace, np.complex)

lmd = [0.001, 1.0e5, 1.05]

# Reconstruct each of the brain images
start = time.time()
# USE NON-LOCAL WITH A h=8
for i, F in enumerate(undersampleF):
    if i == 0:
        recon[i, :, :], firstRecon[i, :, :, :] = iterative.sirt_fft_complex_multi(it, N, F, fractalMine, t, 3, 5, complexOutput=True, h=12, lmd=lmd) # 50
        recon[i, :, :] = fftpack.fftshift(recon[i, :, :])
        print("Image number: ", i)
    
newFirstRecon = np.zeros_like(recon, dtype=np.complex)
for i, ims in enumerate(firstRecon):
    for im in ims:
        newFirstRecon[i, :, :] += (im ** 2)

    newFirstRecon[i, :, :] = fftpack.fftshift(np.abs(np.sqrt(newFirstRecon[i, :, :])))
    
end = time.time()
elapsed = end - start
for i, image in enumerate(images):
    images[i, :, :] = image * 255 / np.max(np.abs(image))
    images[i, :, :] = fftpack.fftshift(image)
print("FFT SIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins")

diff = np.abs(images - recon)
recon = np.abs(recon)
images = np.abs(images)
mse = imageio.immse(images[0,:,:], np.abs(recon[0, :, :]))
ssim = imageio.imssim(np.abs(images[0, :, :]/np.max(np.abs(images[0,:,:]))).astype(float), np.abs(recon[0, :, :]/np.max(np.abs(recon[0,:,:]))).astype(float))
psnr = imageio.impsnr(np.abs(images[0,:,:]), np.abs(recon[0, :, :]))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)


index = 0

plt.figure(1)
plt.imshow(np.abs(fractalMine))

fig = plt.figure(2)
plt.axis('off')
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(np.abs(recon[0, :, :]), cmap='gray', vmin=0, vmax=255)
fig.savefig("knee_fftsirt/test_number_" + str(index)+"reduction_factor_" + '.png', bbox_inches='tight', pad_inches=0)

plt.figure(3)
plt.imshow(np.abs(images[0,:,:]))
plt.figure(9)
plt.imshow(np.abs(diff[0, :, :]))
plt.figure(10)
plt.imshow(np.abs(newFirstRecon[0, :, :]))





# Plot prediction results
fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20,5))
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[3].axis('off')
ax[0].imshow(np.abs(images[0,:,:]), aspect="auto", cmap='gray', vmin=0, vmax=255)
ax[1].imshow(np.abs(fftpack.fftshift(fractalMine)), aspect="auto", cmap='gray', vmin=0, vmax=1)
ax[2].imshow(np.abs(newFirstRecon[0, :, :]), aspect="auto", cmap='gray', vmin=0, vmax=255)
ax[3].imshow(np.abs(recon[0, :, :]), aspect="auto", cmap='gray', vmin=0, vmax=255)
ax[0].set_title("Ground Truth", fontsize=15)
ax[1].set_title("Sampling Pattern", fontsize=15)
ax[2].set_title("Zero Fill", fontsize=15)
ax[3].set_title("Prediction", fontsize=15)
fig.tight_layout()
output_path = "knee_fftsirt/test_number_"+str(index)+'.png'
fig.savefig(output_path)
plt.close(fig)