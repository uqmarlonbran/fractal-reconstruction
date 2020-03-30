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
import filenames
from scipy.io import loadmat, savemat
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

# Images path
imagesPath = "oasis_160/"
randoms = [True]
for bestRandom in randoms:
    if bestRandom:
        fractalsPath = "best_fractals/"
    else:
        fractalsPath = "det_fractals/"
    r = [8]
    
    #create test image
    imageList = filenames.getSortedFileList(imagesPath, "*.png")
    
    ssim = np.zeros((len(r), len(imageList), 1))
    psnr = np.zeros_like(ssim)
    elapsed = np.zeros_like(psnr)
    N = 256
    megaConc = np.zeros((N, N*3))
    
    for j, R in enumerate(r):
        fractalMine = np.array(imageio.imread(fractalsPath + "r" + str(R) + "_best_mask.png"), dtype=np.complex)
        fractalMine = fftpack.ifftshift(fractalMine) // np.max(np.abs(fractalMine))
        
        for i, imagePath in enumerate(imageList):
            
            print("Finished " + str(i) + " out of " + str(len(imageList)) + ".")
            
            image = np.array(imageio.imread(imagesPath + imagePath), dtype=np.complex)
            image = image * 255 / np.max(np.abs(image))
            kspace = fftpack.fft2(image)
            
            [N1, N2] = image.shape
            N = N1
            
            #compute lines
            centered = False
            random = True
            
            if R == 2:
                h = 4
            elif R == 4:
                h = 6
            elif R == 8:
                h = 6
            
            
            undersampleF = np.zeros_like(kspace, np.complex)
            undersampleF = kspace * fractalMine
            
            print("Samples used: ", np.sum(fractalMine) / N**2)
            
            t = (N**2)*1.5 # reduction factor 0.5
            it = 100
            #t = 50 # reduction factor 0.18
            #it = 1250
            
            smoothType = 3
            
            lmd = [0.001, 1.0e5, 5]
            
            # Reconstruct each of the brain images
            start = time.time()
            recon, firstRecon = iterative.sirt_fft_complex(it, N, undersampleF, fractalMine, t, smoothType, 3, h=h, lmd=lmd, metric=False, ground=np.abs(image), insertSamples=True)
            # recon = (recon * 255) / np.max(np.abs(recon))
            
            end = time.time()
            elapsed[j, i] = end - start
            print("FFTSIRT Reconstruction took " + str(elapsed[j, i]) + " secs or " + str(elapsed[j, i]/60) + " mins")  
            mse = metrics.mean_squared_error(np.abs(image), np.abs(recon))
            ssim[j, i] = metrics.structural_similarity(np.abs(image).astype(float), np.abs(recon).astype(float), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
            psnr[j, i] = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
            print("RMSE:", math.sqrt(mse))
            print("SSIM:", ssim[j, i])
            print("PSNR:", psnr[j, i])
            
            diff = np.abs(image - recon)
            
            megaConc[:, 0:N] = np.abs(image)
            megaConc[:, N:N*2] = np.abs(firstRecon)
            megaConc[:, N*2:] = np.abs(recon)
            
            
            im = Image.fromarray(np.abs(megaConc).astype(np.uint8))
            if bestRandom:
                im.save("output_fftsirt_rand_miccai/R" + str(R) + "/test_" + str(i) + '.png')
            else:
                im.save("output_fftsirt_det_miccai/R" + str(R) + "/test_" + str(i) + '.png')
        
        if bestRandom:
            savemat("output_fftsirt_rand_miccai/FFTSIRT_RAND_" + str(R) + "_IMAGES.mat", {'time':elapsed[j, :], 'psnr':psnr[j, :], 'ssim':ssim[j, :], 'R':R, 'mask':fftpack.fftshift(fractalMine)})
        else:
            savemat("output_fftsirt_det_miccai/FFTSIRT_DET_" + str(R) + "_IMAGES.mat", {'time':elapsed[j, :], 'psnr':psnr[j, :], 'ssim':ssim[j, :], 'R':R, 'mask':fftpack.fftshift(fractalMine)})
                                                                                      # index = 0
    
    # plt.figure(1)
    # plt.imshow(np.abs(fractalMine))
    
    # fig = plt.figure(2)
    # plt.axis('off')
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.imshow(np.abs(recon), cmap='gray', vmin=0, vmax=255)
    # # fig.savefig("output_fftsirt_miccai/reduction_factor_" + str(R) + '.png', bbox_inches='tight', pad_inches=0)
    # im = Image.fromarray(np.abs(recon).astype(np.uint8))
    # im.save("output_fftsirt_miccai/reduction_factor_" + str(R) + '.png')
    
    
    # fig = plt.figure(3)
    # plt.axis('off')
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.imshow(np.abs(images), cmap='gray', vmin=0, vmax=255)
    
    
    # fig = plt.figure(4)
    # plt.axis('off')
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.imshow(np.abs(firstRecon), cmap='gray', vmin=0, vmax=255)
    # # fig.savefig("output_fftsirt_miccai/zf_reduction_factor_" + str(R) + '.png', bbox_inches='tight', pad_inches=0)
    # im = Image.fromarray(np.abs(firstRecon).astype(np.uint8))
    # im.save("output_fftsirt_miccai/zf_reduction_factor_" + str(R) + '.png')
    
    # plt.figure(9)
    # plt.imshow(np.abs(diff))
    
    # plt.figure(10)
    # plt.plot(range(0, it), psnrArr)
    # plt.figure(11)
    # plt.plot(range(0, it), ssimArr)



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