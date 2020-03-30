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
from scipy.io import loadmat, savemat
# from fareyFractal import farey_fractal
# from scipy import ndimage 
import matplotlib.pyplot as plt
import scipy.io as io
from makeRandomFractal import makeRandomFractal


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
#load kspace data
mouse=False
if mouse:
    brain_dict = io.loadmat('data/shrew_Cartesian.mat')
    kspace = brain_dict.get('Cartesian_data')
    images = fftpack.fftshift(fftpack.ifft2(kspace))
    images = images / np.max(np.abs(images))
    images *= 255
    kspace = fftpack.fft2(images)
else:   
    brain_dict = io.loadmat('data/Cartesian_LEGO.mat')
    kspace = brain_dict.get('Cartesian_data')
    # Centre the image
    kspace = fftpack.ifftshift(kspace)
    kspace = np.roll(kspace, 1, axis=0) #fix 1 pixel shift
    images = fftpack.fftshift(fftpack.ifft2(kspace))
    images = images / np.max(np.abs(images))
    images *= 255
    kspace = fftpack.fft2(images)


[N, N1] = images.shape
# fftkSpace = kspace/(np.abs(kspace)).max()
# sampling_mask = brain_dict.get('mask')
# sampling_pdf = brain_dict.get('pdf')
#load Cartesian data
#Attention: You must ensure the kspace data is correctly centered or not centered.
#-------------------------------


#-------------------------------
#compute lines
centered = False
random = True
cartesian = True
r = [2, 4, 6, 8, 10]
N1 = len(r)
reduc = np.zeros((N1, 1))
mse = np.zeros((N1, 1))
psnr = np.zeros((N1, 1))
ssim = np.zeros((N1, 1))
elapsed = np.zeros((N1, 1))
recons = np.zeros((N1, N, N), dtype=np.complex)
firstRecons = np.zeros_like(recons)
for i, R in enumerate(r):
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
    
    if not random:
        lines, angles, \
            mValues, sampling_mask, \
            oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', 
                                                twoQuads=True)
        lines = np.array(lines)
    else:
        sampling_mask, samplingFilter = makeRandomFractal(N, (1/R) * 0.9, tilingSize=tilingSize, withTiling=True)
        reduc[i] = np.sum(sampling_mask) / (N**2)
    if cartesian:
        sampling_mask, r_factor= iterative.generate_mask_alpha(size=[N,N], r_factor_designed=R, r_alpha=2,seed=-1)
        reduc[i] = r_factor
        
    fractalMine = sampling_mask
    # Sample kspace
    undersampleF = kspace * fractalMine
    
    print("Samples used: ", R)
    
    t = (N**2)*1.5 # reduction factor 0.5
    # it = 50
    # it = 150
    it =149
    
    smoothType = 3
    
    lmd = [0.004, 1.0e5, 5]
    
    # Reconstruct each of the brain images
    start = time.time()
    #3 is best for shrew and 2 for the lego
    recons[i, :, :], firstRecons[i, :, :], psnrArr, ssimArr = iterative.sirt_fft_complex(it, N, undersampleF, fractalMine, t, smoothType, 3, h=8, lmd=lmd, metric=True, ground=np.abs(images), insertSamples=False)
    # recon = (recon) / np.max(np.abs(recon))
    # recon *= 255
    recon = recons[i, :, :]
    firstRecon = firstRecons[i, :, :]
    end = time.time()
    elapsed[i] = end - start
    print("FFTSIRT Reconstruction took " + str(elapsed[i]) + " secs or " + str(elapsed[i]/60) + " mins")  
    mse[i] = metrics.mean_squared_error(np.abs(images), np.abs(recon))
    ssim[i] = metrics.structural_similarity(np.abs(images).astype(float), np.abs(recon).astype(float), data_range=255)
    psnr[i] = metrics.peak_signal_noise_ratio(np.abs(images), np.abs(recon), data_range=255)
    print("RMSE:", math.sqrt(mse[i]))
    print("SSIM:", ssim[i])
    print("PSNR:", psnr[i])


if cartesian:
    savemat("FFTSIRT_Cartesian_LEGO.mat", {'time':elapsed, 'mse':mse, 'psnr':psnr, 'ssim':ssim, 'r':r, 'actual_r':reduc})
else:
    savemat("FFTSIRT_Fractal_LEGO.mat", {'time':elapsed, 'mse':mse, 'psnr':psnr, 'ssim':ssim, 'r':r, 'actual_r':reduc})

# diff = np.abs(images - recon)

# index = 0

# plt.figure(1)
# plt.imshow(np.abs(fractalMine))

# fig = plt.figure(2)
# plt.axis('off')
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.imshow(np.abs(recon), cmap='gray')
# # fig.savefig("output_fftsirt/test_number_" + str(index)+"reduction_factor_" + str(R) + '.png', bbox_inches='tight', pad_inches=0)


# fig = plt.figure(3)
# plt.axis('off')
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.imshow(np.abs(images), cmap='gray')

# plt.figure(4)
# plt.imshow(np.abs(firstRecon))
# plt.figure(9)
# plt.imshow(np.abs(diff), cmap='gray')


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
# fig.savefig(output_path)
# plt.close(fig)