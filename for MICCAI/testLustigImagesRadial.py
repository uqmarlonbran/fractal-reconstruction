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
#-------------------------------
# Get Cartesian co-ords for polar co-ords
def get_x_y(rad, ang, N):
    realRad = rad - N // 2
    y = np.sin(ang * np.pi / 180) * realRad
    x = np.cos(ang * np.pi / 180) * realRad
    return x, y

imagesPath = "oasis_160/"

imageList = filenames.getSortedFileList(imagesPath, "*.png")

N = 512

#compute lines
centered = False
outPath = "output_nlcg_rad_miccai/"
outStub = "NLCG_rad"
r = [2, 4, 8]
ssim = np.zeros((len(r), len(imageList), 1))
psnr = np.zeros_like(ssim)
elapsed = np.zeros_like(psnr)
megaConc = np.zeros((N, N*3))



# Create the cartesian grid
grid_x, grid_y = np.mgrid[-N//2:N//2-1:N*1j, -N//2:N//2-1:N*1j] # grid with N samples, range defines rotation

for j, R in enumerate(r):
    N = 256
    # Golden angle
    nums = N // R
    # angleDelta = 137.507764
    angleDelta = 180 / nums
    # Generate the sampling masks
    angleArray = np.array([0], dtype=np.float)
    while len(angleArray) < N // R:
        angleArray = np.append(angleArray, [angleArray[-1] + angleDelta])
        
    print(len(angleArray))
    # N = 600
    # Get locations for data points
    N = 512
    polarLoc = np.zeros((N * (N // R), 2), dtype=np.float)
    polarVals = np.zeros((N * (N // R)), dtype=np.complex)
    
    rads = np.array(range(0, N))
    fig = plt.figure(frameon=False, figsize=[8, 8])
    ax = plt.axes()
    # plt.axis('off')
    plt.margins(0,0)
    
    
    for th, theta in enumerate(angleArray):
        polarLoc[th*N:th*N + N, :] = np.array(get_x_y(rads, theta, N)).T
        plt.plot(polarLoc[th*N:th*N + N, 0], polarLoc[th*N:th*N + N, 1], 'w', linewidth=1)
    
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ax.set_facecolor("black") # Black
    N=256
    plt.axis([-N//2, N//2, -N//2, N//2])  
    plt.show()
    fig.savefig("knee_radial/_sampling_mask_R" + str(R) + '.png', bbox_inches='tight', pad_inches=0)
    fig.savefig("knee_radial/_sampling_mask_R" + str(R) + '.eps', bbox_inches='tight', pad_inches=0)
    # for i, imagePath in enumerate(imageList):
    #     print("Finished " + str(i) + " out of " + str(len(imageList)) + ".")
    #     #load kspace data
    #     image = np.array(imageio.imread(imagesPath + imagePath), dtype=np.complex)
        
    #     sinogram = radon(np.abs(image), theta=angleArray, circle=True)
    #     sinogramDFT = fftpack.fftshift(fftpack.fft(sinogram, axis=0))
        
    #     reconstruction_fbp = iradon(sinogram, theta=angleArray, circle=True)
    #     reconstruction_art = iradon_sart(sinogram, theta=angleArray)
    #     reconstruction_art = iradon_sart(sinogram, theta=angleArray, image=reconstruction_art)
    #     reconstruction_art = iradon_sart(sinogram, theta=angleArray, image=reconstruction_art)
        
    #     for th, theta in enumerate(angleArray):
    #         polarVals[th*N:th*N + N] = sinogramDFT[rads, th]
            
    #     interp_dft = np.squeeze(griddata(polarLoc, polarVals, (grid_x, grid_y), method='linear', fill_value=0.0))
    #     dft = fftpack.ifftshift(interp_dft)
    #     interp = fftpack.ifft2(interp_dft)
    #     break
    
    # plt.imshow(reconstruction_fbp)
    # plt.figure()
    # plt.imshow(reconstruction_art)
    # plt.figure()
    # plt.imshow(np.abs(interp))
    # plt.figure()
    # plt.imshow(np.abs(sinogramDFT))
    # plt.figure()
    # plt.imshow(np.abs(interp_dft))
    # plt.figure()
    # imdft = fftpack.fftshift(fftpack.fft2(image))
    # plt.imshow(np.abs(imdft))
    
    # recon1 = reconstruction_fbp
    # recon2 = reconstruction_art
    
    # ssim1 = metrics.structural_similarity(np.abs(image), np.abs(recon1), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    # psnr1 = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon1), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    # ssim2 = metrics.structural_similarity(np.abs(image), np.abs(recon2), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    # psnr2 = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon2), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    
    # break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #     dftSpace = fftpack.fftshift(fftpack.ifft2(image)) 
    #     powSpect = np.abs(dftSpace)
    #     fftkSpaceShifted = dftSpace/(np.abs(powSpect)).max()
    #     image = fftpack.ifft2(fftkSpaceShifted)
    #     kSpace = fftkSpaceShifted*sampling_mask    
        
    #     #Define parameter class for nlcg
    #     params = param_class.nlcg_param()
    #     params.FTMask = sampling_mask
    #     params.TVWeight = 0.0002
    #     params.wavWeight = 0.0002
    #     iterations = 8
    #     params.data = kSpace
        
    #     recon = np.zeros_like(fftkSpaceShifted, np.complex)
        
    #     start = time.time() #time generation
    #     zf_image = cs_utils.ifft2u(kSpace)
    #     #zf_image = cs_utils.ifft2u(kSpace/sampling_pdf)
        
    #     wavelet_x0 = cs_utils.dw2(zf_image)
    #     wavelet_x0_coeff = wavelet_x0.coeffs
    #     wavelet_x0_coeffabs = np.abs(wavelet_x0_coeff)
        
    #     #compute reconstruction
    #     wavelet_x = cs_utils.dw2(zf_image)
    #     params.data = kSpace
    #     for k in range(1, iterations):
    #         wavelet_x = fnlCg(wavelet_x, params)
    #         recon = cs_utils.idw2(wavelet_x)
    #     # recon = recon * 255 / np.max(np.abs(recon))
    #     recon /= N
        
    #     zf_image = zf_image * 255 / np.max(np.abs(zf_image))
        
    #     print("Done")
    #     end = time.time()
        
    #     elapsed[j, i] = end - start
    #     print("CS Reconstruction took " + str(elapsed[j, i]) + " secs or " + str(elapsed[j, i]/60) + " mins")  
    #     mse = metrics.mean_squared_error(np.abs(image), np.abs(recon))
    #     ssim[j, i] = metrics.structural_similarity(np.abs(image).astype(float), np.abs(recon).astype(float), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    #     psnr[j, i] = metrics.peak_signal_noise_ratio(np.abs(image), np.abs(recon), data_range=np.max(np.abs(image)) - np.min(np.abs(image)))
    #     print("RMSE:", math.sqrt(mse))
    #     print("SSIM:", ssim[j, i])
    #     print("PSNR:", psnr[j, i])
        
    #     diff = np.abs(image) - np.abs(recon)
        
    #     image = image * 255 / np.max(np.abs(image))
    #     recon = recon * 255 / np.max(np.abs(recon))
        
    #     megaConc[:, 0:N] = np.abs(image)
    #     megaConc[:, N:N*2] = np.abs(zf_image)
    #     megaConc[:, N*2:] = np.abs(recon)
        
    #     im = Image.fromarray(np.abs(megaConc).astype(np.uint8))
    #     im.save(outPath + "R" + str(R) + "/test_" + str(i) + '.png')
        
    # # Save statistics
    # savemat(outPath + outStub + str(R) + "_IMAGES.mat", {'time':elapsed[j, :], 'psnr':psnr[j, :], 'ssim':ssim[j, :], 'R':R, 'mask':sampling_mask})



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
    
