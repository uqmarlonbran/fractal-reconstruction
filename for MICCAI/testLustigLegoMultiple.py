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
import matplotlib.pyplot as plt
import scipy.io as io
import imageio as imageio 
from PIL import Image
from makeRandomFractal import makeRandomFractal
import finitetransform.samplers as samplers
# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

def create_sampler(mode, M=256, r=0.5, r_alpha=2, acs=0., seed=-1, 
                       N=256, k=1, K=0.1, s=8, twoQuads=True):
    if 'cart' in mode.lower():
        params = {'M': M, 'r': r, 'r_alpha': r_alpha, 'acs': acs}
        if '1d' in mode.lower():
            sampler = samplers.OneDimCartesianRandomSampler(r=r, r_alpha=r_alpha, acs=acs, seed=-1)
        elif '2d' in mode.lower():
            sampler = samplers.TwoDimCartesianRandomSampler(r=r, r_alpha=r_alpha, acs=acs, seed=-1)
    elif 'fractal' in mode.lower():
        params = {'N': N, 'k': k, 'M': M, 'K': K, 'r': r, 's': s,
                'ctr': acs, 'twoQuads': twoQuads}
        sampler = samplers.FractalSampler(k, K, r, acs, s, seed=-1)
    return sampler, params

#parameters
floatType = np.complex
#-------------------------------
#load kspace data
mouse=False
if mouse:
    brain_dict = io.loadmat('data/shrew_Cartesian.mat')
else:   
    brain_dict = io.loadmat('data/Cartesian_LEGO.mat')
    
dftSpace = brain_dict.get('Cartesian_data')
powSpect = np.abs(dftSpace)
fftkSpaceShifted = dftSpace/(np.abs(powSpect)).max()
image = fftpack.fftshift(fftpack.ifft2(fftkSpaceShifted))
fftkSpaceShifted = fftpack.fft2(image)
# image = fftpack.fftshift(image) * (np.abs(powSpect)).max()

[N, N1] = image.shape
#-------------------------------
#compute lines
centered = False
random = True
r = [2, 4, 8]
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
        outPath = "output_nlcg_1d_lego/"
        fractalsPath = "cart1d/"
        outStub = "NLCG_1D_R"
    else:
        outPath = "output_nlcg_2d_lego/"
        fractalsPath = "cart2d/"
        outStub = "NLCG_2D_R"
    r = [2, 4, 8]
    ssim = np.zeros((len(r)))
    psnr = np.zeros_like(ssim)
    elapsed = np.zeros_like(psnr)
    megaConc = np.zeros((N, N*3))
    for j, R in enumerate(r):
        
        fractalMine = np.array(imageio.imread(fractalsPath + "r" + str(R) + "_best_mask.png"), dtype=np.complex)
        fractalMine = fftpack.ifftshift(fractalMine) // np.max(np.abs(fractalMine))
        sampling_mask = fftpack.fftshift(fractalMine)
        
        # print("Samples used: ", R)
        kSpace = fftkSpaceShifted*sampling_mask    
        
        #Define parameter class for nlcg
        # 
        params = param_class.nlcg_param()
        params.FTMask = sampling_mask
        params.TVWeight = 0.0004
        params.wavWeight = 0.0004
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
        recon /= N
        # recon = fftpack.fftshift(recon)
        # zf_image = fftpack.fftshift(zf_image)
        zf_image = zf_image * 255 / np.max(np.abs(zf_image))
        
        print("Done")
        end = time.time()
        elapsed[j] = end - start
        print("CS Reconstruction took " + str(elapsed[j]) + " secs or " + str(elapsed[j]/60) + " mins")  
        mse = metrics.mean_squared_error(np.abs(image)*255/np.max(np.abs(image)), np.abs(recon)*255/np.max(np.abs(recon)))
        ssim[j] = metrics.structural_similarity(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
        psnr[j] = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
        print("RMSE:", math.sqrt(mse))
        print("SSIM:", ssim[j])
        print("PSNR:", psnr[j])
        
        diff = np.abs(image) - np.abs(recon)
        
        imageSave = image * 255 / np.max(np.abs(image))
        recon = recon * 255 / np.max(np.abs(recon))
        
        megaConc[:, 0:N] = np.abs(imageSave)
        megaConc[:, N:N*2] = np.abs(zf_image)
        megaConc[:, N*2:] = np.abs(recon)
        
        im = Image.fromarray(np.abs(megaConc).astype(np.uint8))
        im.save(outPath + "R" + str(R) + "/lego" '.png')
            
        # Save statistics
        savemat(outPath + outStub + str(R) + "_LEGO.mat", {'time':elapsed[j], 'psnr':psnr[j], 'ssim':ssim[j], 'R':R, 'mask':sampling_mask})
            
# diff = np.abs(images) - np.abs(recon)

# plt.figure(1)
# plt.imshow(np.abs(sampling_mask))

# fig = plt.figure(2)
# plt.axis('off')
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.imshow(np.abs(recon), cmap='gray', vmin=0, vmax=255)

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
# plt.imshow(np.abs(zf_image), cmap='gray', vmin=0, vmax=255)

# plt.figure(9)
# plt.imshow(np.abs(diff))

# if fractal:
#     savemat("LUSTIG_Cartesian_LEGO.mat", {'time':elapsed, 'mse':mse, 'psnr':psnr, 'ssim':ssim, 'r':r, 'actual_r':reduc})
# else:
#     savemat("LUSTIG_Fractal_LEGO.mat", {'time':elapsed, 'mse':mse, 'psnr':psnr, 'ssim':ssim, 'r':r, 'actual_r':reduc})
