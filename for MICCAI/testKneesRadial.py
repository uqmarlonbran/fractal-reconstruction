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
import filenames
import matplotlib.pyplot as plt
import scipy.io as io
from makeRandomFractal import makeRandomFractal
from PIL import Image
import imageio as imageio 
#cs
import param_class
import cs_utils
from fnlCg import fnlCg
# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

from skimage.transform import radon, iradon, iradon_sart
from scipy.interpolate import griddata

#parameters
floatType = np.complex

# load Cartesian data
dftSpace, num_cases = loadNii.load_data_channels("knees/knee_kspace_slices_reduced")
dftSpace = np.squeeze(dftSpace)
images = fftpack.fftshift(fftpack.ifft2(dftSpace))
dftSpace = np.zeros_like(images, dtype=np.complex)
imMax = np.max(np.abs(images))
#Attention: You must ensure the kspace data is correctly centered or not centered.
for i, im in enumerate(images):
    dftSpace[i, :, :] = fftpack.fft2(fftpack.ifftshift(im)) * 255 / imMax
    dftSpace[i, :, :] = np.roll(dftSpace[i, :, :], 1, axis=0)

images = fftpack.ifft2(dftSpace)

#Attention: You must ensure the kspace data is correctly centered or not centered.

dftSpace = fftpack.fft2(images)
dftSpace = dftSpace[0,:,:]
dftSpace = dftSpace / np.max(np.abs(dftSpace))
image = fftpack.ifftshift(fftpack.ifft2(dftSpace))
# dftSpace = fftpack.fftshift(fftpack.fft2(image * 255 / np.max(np.abs(image))))
dftSpace = fftpack.fftshift(fftpack.fft2(image * 255 / np.max(np.abs(image))))
image = fftpack.ifft2(dftSpace)
fftkSpaceShifted = dftSpace

im = Image.fromarray(np.rot90(np.abs(image*255/np.max(np.abs(image))), k=3).astype(np.uint8))
im.save("knee_radial/knee" + '.png')

N = 320

#compute lines
centered = False
outPath = "knee_radial/"
outStub = "sart_rad"
r = [2, 4, 6, 8]
ssim = np.zeros((len(r)))
psnr = np.zeros_like(ssim)
rmse = np.zeros_like(psnr)
elapsed = np.zeros_like(psnr)

# Golden angle

iterations = 15
for j, R in enumerate(r):
    nums = N // R
    # angleDelta = 137.507764
    angleDelta = 180 / nums
    # Generate the sampling masks
    angleArray = np.array([0], dtype=np.float)
    while len(angleArray) < N // R:
        angleArray = np.append(angleArray, [angleArray[-1] + angleDelta])
    #load kspace data
    
    sinogram = radon(np.abs(image), theta=angleArray, circle=False)   
    reconstruction_fbp = iradon(sinogram, theta=angleArray, circle=False)
    start = time.time()
    for it in range(0, iterations):
        if it == 0:
            reconstruction_art = iradon_sart(sinogram, theta=angleArray)
        else:
            reconstruction_art = iradon_sart(sinogram, theta=angleArray, image=reconstruction_art)
        
    [a, b] = reconstruction_art.shape
    recon = reconstruction_art[a//2-N//2:a//2+N//2,b//2-N//2:b//2+N//2]
    
    zf_image = reconstruction_fbp * 255 / np.max(np.abs(reconstruction_fbp))
    
    print("Done")
    end = time.time()
    
    elapsed[j] = end - start
    print("CS Reconstruction took " + str(elapsed[j]) + " secs or " + str(elapsed[j]/60) + " mins")  
    rmse[j] = np.sqrt(metrics.mean_squared_error(np.abs(image), np.abs(recon)))
    ssim[j] = metrics.structural_similarity(np.abs(image).astype(float), np.abs(recon).astype(float), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    psnr[j] = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    print("RMSE:", rmse)
    print("SSIM:", ssim)
    print("PSNR:", psnr)   
    diff = np.abs(image) - np.abs(recon)
    
    image = image * 255 / np.max(np.abs(image))
    recon = recon * 255 / np.max(np.abs(recon))
    
    im = Image.fromarray(np.rot90(np.abs(zf_image), k=3).astype(np.uint8))
    im.save(outPath + "zf_" + "R" + str(R) + '.png')
    im = Image.fromarray(np.rot90(np.abs(recon), k=3).astype(np.uint8))
    im.save(outPath + outStub + "_R" + str(R) + '.png')
        
# Save statistics
savemat(outPath + outStub + "_KNEES.mat", {'time':elapsed, 'psnr':psnr, 'ssim':ssim, 'R':r, 'rmse':rmse, 'angles':angleArray})



# plt.figure(1)
# plt.imshow(np.abs(sampling_mask))

# fig = plt.figure(2)
# plt.axis('off')
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.imshow(np.abs(recon), cmap='gray')

# fig = plt.figure(3)
# plt.axis('off')
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.imshow(np.abs(image), cmap='gray')

# fig = plt.figure(4)
# plt.axis('off')
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.imshow(np.abs(zf_image), cmap='gray')

# plt.figure(9)
# plt.imshow(np.abs(diff))

# megaConc = np.zeros((N, N*3))
# megaConc[:, 0:N] = np.abs(image) * 255 / np.max(np.abs(image))
# megaConc[:, N:N*2] = np.abs(zf_image) * 255 / np.max(np.abs(zf_image))
# megaConc[:, N*2:] = np.abs(recon) * 255 / np.max(np.abs(recon))


# im = Image.fromarray(np.abs(megaConc).astype(np.uint8))
# im.save("cs/reduction_factor_" + str(R) + '.png')

# # if cartesian:
# #     savemat("LUSTIG_Cartesian_LEGO.mat", {'time':elapsed, 'mse':mse, 'psnr':psnr, 'ssim':ssim, 'r',:r})
# # else:
    
