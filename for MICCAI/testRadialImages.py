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

imagesPath = "oasis_160/"

imageList = filenames.getSortedFileList(imagesPath, "*.png")
outPath = "radialImages/"
N = 256
outStub = "RADIAL"
#compute lines
r = [2, 4, 8]
r = [4, 8]
ssim = np.zeros((len(r), len(imageList), 1))
psnr = np.zeros_like(ssim)
rmse = np.zeros_like(psnr)
elapsed = np.zeros_like(psnr)
iterations = 15
for j, R in enumerate(r):
    for i, imagePath in enumerate(imageList):
        megaConc = np.zeros((N, N*3))
        nums = N // R
        # angleDelta = 137.507764
        angleDelta = 180 / nums
        # Generate the sampling masks
        angleArray = np.array([0], dtype=np.float)
        while len(angleArray) < N // R:
            angleArray = np.append(angleArray, [angleArray[-1] + angleDelta])
            
            
        #load kspace data
        image = np.array(imageio.imread(imagesPath + imagePath))
        image = image / np.max(image)
        image *= 255
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
        
        elapsed[j, i] = end - start
        print("CS Reconstruction took " + str(elapsed[j, i]) + " secs or " + str(elapsed[j, i]/60) + " mins")  
        rmse[j, i] = np.sqrt(metrics.mean_squared_error(np.abs(image), np.abs(recon)))
        ssim[j, i] = metrics.structural_similarity(np.abs(image).astype(float), np.abs(recon).astype(float), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
        psnr[j, i] = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
        print("RMSE:", rmse[j, i])
        print("SSIM:", ssim[j, i])
        print("PSNR:", psnr[j, i]) 
        diff = np.abs(image) - np.abs(recon)
        
        image = image * 255 / np.max(np.abs(image))
        recon = recon * 255 / np.max(np.abs(recon))
        
        megaConc[:, 0:N] = np.abs(image)
        megaConc[:, N:N*2] = np.abs(zf_image)
        megaConc[:, N*2:] = np.abs(recon)
        
        im = Image.fromarray(np.abs(megaConc).astype(np.uint8))
        im.save(outPath + "R" + str(R) + "/test_" + str(i) + '.png')
            
    # Save statistics
    savemat(outPath + outStub + str(R) + "_IMAGES.mat", {'time':elapsed[j, :], 'psnr':psnr[j, :], 'ssim':ssim[j, :], 'R':R})



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
    
