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

#parameters
floatType = np.complex
#-------------------------------

# image = images * 255 / np.max(np.abs(images))

# fractalsPath = "det_fractals/"

imagesPath = "oasis_160/"

imageList = filenames.getSortedFileList(imagesPath, "*.png")

N = 256

#compute lines
centered = False
bestRandom = True
cart1D = False
fracts = [True, False]
carts = [True, False]
# if bestRandom and not cart1D:
#     outPath = "output_nlcg_rand_miccai/"
#     fractalsPath = "best_fractals/"
#     outStub = "NLCG_RAND"
# else:
#     outPath = "output_nlcg_det_miccai/"
#     fractalsPath = "det_fractals/"
#     outStub = "NLCG_DET"
for cart1D in carts:
    if cart1D:
        outPath = "output_nlcg_1d_miccai/"
        fractalsPath = "cart1d/"
        outStub = "NLCG_1D"
    else:
        outPath = "output_nlcg_2d_miccai/"
        fractalsPath = "cart2d/"
        outStub = "NLCG_2D"
    r = [2, 4, 8]
    ssim = np.zeros((len(r), len(imageList), 1))
    psnr = np.zeros_like(ssim)
    elapsed = np.zeros_like(psnr)
    megaConc = np.zeros((N, N*3))
    # print("Samples used: ", R)
    for j, R in enumerate(r):
        fractalMine = np.array(imageio.imread(fractalsPath + "r" + str(R) + "_best_mask.png"), dtype=np.complex)
        fractalMine = fftpack.ifftshift(fractalMine) // np.max(np.abs(fractalMine))
        sampling_mask = fftpack.fftshift(fractalMine)
        
        for i, imagePath in enumerate(imageList):
            print("Finished " + str(i) + " out of " + str(len(imageList)) + ".")
            #load kspace data
            image = np.array(imageio.imread(imagesPath + imagePath), dtype=np.complex)
            dftSpace = fftpack.fftshift(fftpack.ifft2(image)) 
            powSpect = np.abs(dftSpace)
            fftkSpaceShifted = dftSpace/(np.abs(powSpect)).max()
            image = fftpack.ifft2(fftkSpaceShifted)
            kSpace = fftkSpaceShifted*sampling_mask    
            
            #Define parameter class for nlcg
            params = param_class.nlcg_param()
            params.FTMask = sampling_mask
            params.TVWeight = 0.0002
            params.wavWeight = 0.0002
            iterations = 8
            params.data = kSpace
            
            recon = np.zeros_like(fftkSpaceShifted, np.complex)
            
            start = time.time() #time generation
            zf_image = cs_utils.ifft2u(kSpace)
            #zf_image = cs_utils.ifft2u(kSpace/sampling_pdf)
            
            wavelet_x0 = cs_utils.dw2(zf_image)
            wavelet_x0_coeff = wavelet_x0.coeffs
            wavelet_x0_coeffabs = np.abs(wavelet_x0_coeff)
            
            #compute reconstruction
            wavelet_x = cs_utils.dw2(zf_image)
            params.data = kSpace
            for k in range(1, iterations):
                wavelet_x = fnlCg(wavelet_x, params)
                recon = cs_utils.idw2(wavelet_x)
            # recon = recon * 255 / np.max(np.abs(recon))
            recon /= N
            
            zf_image = zf_image * 255 / np.max(np.abs(zf_image))
            
            print("Done")
            end = time.time()
            
            elapsed[j, i] = end - start
            print("CS Reconstruction took " + str(elapsed[j, i]) + " secs or " + str(elapsed[j, i]/60) + " mins")  
            rmse[j, i] = np.sqrt(metrics.mean_squared_error(np.abs(image), np.abs(recon)))
            ssim[j, i] = metrics.structural_similarity(np.abs(image).astype(float), np.abs(recon).astype(float), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
            psnr[j, i] = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
            print("RMSE:", math.sqrt(rmse))
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
        savemat(outPath + outStub + str(R) + "_IMAGES.mat", {'time':elapsed[j, :], 'psnr':psnr[j, :], 'ssim':ssim[j, :], 'R':R, 'mask':sampling_mask})



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
    
