# -*- coding: utf-8 -*-
"""
Create a finite slice sampling of k-space and reconstruct using MLEM

Created on Wed Jul 13 21:07:06 2016

#Lego Sparse: K=0.8 (64 slices, 50% data), s=12, i=301, h=0.8, N=128 NLM 
#Lego High Fidelity: K=1.0 (72 slices), s=12, i=1501, h=0.5, N=128 NLM 

@author: shakes
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './fareyFunctions')
from phantominator import shepp_logan
import finitetransform.imageio as imageio #local module
import finitetransform.numbertheory as nt #local modules
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, denoise_bilateral
import scipy.fftpack as fftpack
import pyfftw
import numpy as np
import finite as finite
import time
import math
from fareyFractal import farey_fractal, normalizer

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
from matplotlib.backends.backend_pdf import PdfPages

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

#parameters
N = 257 #N = 128, 200. N = 256, 400
M = N
K = 0.4
s = 12 #0.5, 8; 0.8, 12; 1.0, 12; 1.2, 16; 1.5, 22; 2, 16
iterations = 300
floatType = np.complex
twoQuads = True
plotColourBar = True
plotSampling = True
smoothReconMode = 0 #0-None,1-TV,2-NL,3-Median, 4-Bilateral
smoothIncrement = 10
#smoothMidIteration = iterations/2.0
#smoothMaxIteration = iterations-10
smoothMidIteration = iterations-8*smoothIncrement
smoothMaxIteration = iterations-4*smoothIncrement
print("N:", N, "M:", M, "s:", s, "i:", iterations)
p = nt.nearestPrime(N)
p = N
#bounds
lValue = -150
lBound = complex(lValue, lValue)
uValue = 150
uBound = complex(uValue, uValue)
BL = np.full((p, p), lBound, dtype=floatType) #p or N?
BU = np.full((p, p), uBound, dtype=floatType)


#-------------------------------
#load kspace data
from scipy.io import loadmat

#load Cartesian data
#Attention: You must ensure the kspace data is correctly centered or not centered.
#x = loadmat('data/phantom_water_4.mat')
x = loadmat('data/Cartesian_LEGO.mat')
data_key = 'Cartesian_data'
image_key = 'image'
image = x[image_key]
image = shepp_logan(N)
#image = image * uValue/np.max(np.max(image))
kspace = fftpack.fftshift(fftpack.fft2(image))
#kspace = x[data_key]
#kspace = fftpack.fftshift(kspace)
print("kSpace Shape:", kspace.shape)
kMaxValue = np.max(kspace)
kMinValue = np.min(kspace)
print("k-Space Max Value:", kMaxValue)
print("k-Space Min Value:", kMinValue)
print("k-Space Max Magnitude:", np.abs(kMaxValue))
print("k-Space Min Magnitude:", np.abs(kMinValue))

#-------------------------------
#compute the Cartesian reconstruction for comparison
print("Computing Chaotic Reconstruction...")
dftSpace = kspace
image = fftpack.ifft2(dftSpace) #the '2' is important
#image = fftpack.ifftshift(image)
image = np.abs(image)
maxValue = np.max(image)
minValue = np.min(image)

#dftSpace = np.roll(kspace, 1, axis=0) #fix 1 pixel shift
print("Image Max Value:", maxValue)
print("Image Min Value:", minValue)


#-------------------------------
#compute lines
centered = True

# Generate the fractal
lines, angles, mValues, fractalMine, R = farey_fractal(N, 10, centered=centered)
#lines, mValues = finite.computeLines(dftSpace, angles, centered, twoQuads)
lines = np.array(lines)
#samples used
sampleNumber = lines.shape[0] * N
print("Samples used:", sampleNumber, ", proportion:", sampleNumber/float(N*N))
print("Lines proportion:", lines.shape[0]/float(N))
#-------------
# Measure finite slice
from scipy import ndimage
print("Measuring slices")
powSpectGrid = np.abs(dftSpace)
if N%2==0:
    projs = int(N+N/2)
else:
    projs = N+1
drtSpace = np.zeros((projs, p), floatType)
for i, line in enumerate(lines):
    u, v = line
    sliceReal = ndimage.map_coordinates(np.real(dftSpace), [u,v])
    sliceImag = ndimage.map_coordinates(np.imag(dftSpace), [u,v])
    slice = sliceReal+1j*sliceImag
    finiteProjection = (fftpack.ifft(slice)) # recover projection using slice theorem
    drtSpace[mValues[i],:] = finiteProjection

#recon = finite.ifrt_complex(drtSpace, N, mValues=mValues)
#drtSpace = finite.frt_complex(recon, N)
#-------------------------------
#define ABMLEM    
def abmlem_expand(iterations, p, g_j, os_mValues, projector, backprojector, epsilon=1e3, dtype=np.int32):
    '''
    # Shakes' implementation
    # From Lalush and Wernick;
    # f^\hat <- (f^\hat / |\sum h|) * \sum h * (g_j / g)          ... (*)
    # where g = \sum (h f^\hat)                                   ... (**)
    #
    # self.f is the current estimate f^\hat
    # The following g from (**) is equivalent to g = \sum (h f^\hat)
    '''

    center = False
    fdtype = floatType
    f = np.ones((p,p), fdtype)
    
    for i in range(0, iterations):
        print("Iteration:", i)
#            print("Subset:", j)
        muFinite = len(mValues)
        
        fL = f - BL
        fU = BU - f
        
        L = projector(BL, p, fdtype, mValues)
        U = projector(BU, p, fdtype, mValues)
        gL = projector(fL, p, fdtype, mValues)
        gU = projector(fU, p, fdtype, mValues)
    
        # form parenthesised term (g_j / g) from (*)
        rL = np.copy(g_j - L)
        rLReal = np.real(rL)
        rLImag = np.imag(rL)
        rU = np.copy(U - g_j)
        rUReal = np.real(rU)
        rUImag = np.imag(rU)
        for m in mValues:
            for y in range(p):
                rLReal[m,y] /= np.real(gL[m,y])
                rLImag[m,y] /= np.imag(gL[m,y])
                rUReal[m,y] /= np.real(gU[m,y])
                rUImag[m,y] /= np.imag(gU[m,y])
    
        # backproject to form \sum h * (g_j / g)
        g_rLReal = backprojector(rLReal, p, norm, center, 1, 0, mValues) / muFinite
        g_rLImag = backprojector(rLImag, p, norm, center, 1, 0, mValues) / muFinite
        g_rUReal = backprojector(rUReal, p, norm, center, 1, 0, mValues) / muFinite
        g_rUImag = backprojector(rUImag, p, norm, center, 1, 0, mValues) / muFinite
    
        # Combine the upper/lower bounds and complex parts
        IL = np.real(fL)*g_rLReal + 1j*np.imag(fL)*g_rLImag
        IU = np.real(fU)*g_rUReal + 1j*np.imag(fU)*g_rUImag
        
        #combine to get f
        f = (IL*BU + IU*BL) / (IL + IU)
        
        if smoothReconMode > 0 and i % smoothIncrement == 0 and i > 0: #smooth to stem growth of noise
            fCenter = fftpack.ifftshift(f) #avoid padding issues with some smoothing algorithms by ensuring image is centered
            fReal = np.real(fCenter)
            fImag = np.imag(fCenter)
            if smoothReconMode == 1:
                print("Smooth TV")
                fReal = denoise_tv_chambolle(fReal, 0.02, multichannel=False)
                fImag = denoise_tv_chambolle(fImag, 0.02, multichannel=False)
            elif smoothReconMode == 2:
                h = 1.5
                if i > smoothMaxIteration:
                    h /= 2.0
                print("Smooth NL h:",h)
                fReal = denoise_nl_means(fReal, patch_size=3, patch_distance=7, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                fImag = denoise_nl_means(fImag, patch_size=3, patch_distance=7, h=h, multichannel=False, fast_mode=True).astype(fdtype)
            elif smoothReconMode == 3:
                print("Smooth Median")
                fReal = ndimage.median_filter(fReal, 3)
                fImag = ndimage.median_filter(fImag, 3)
            elif smoothReconMode == 4:
                print("Smooth Bilateral")
                fReal = denoise_bilateral(fReal, sigma_color=0.15, sigma_spatial=7, multichannel=False).astype(fdtype)
                fImag = denoise_bilateral(fImag, sigma_color=0.15, sigma_spatial=7, multichannel=False).astype(fdtype)
                
            f = fftpack.fftshift(fReal +1j*fImag)
        
    return f

def abmlem_expand_complex(iterations, p, g_j, os_mValues, projector, backprojector, epsilon=1e3, dtype=np.int32):
    '''
    # Shakes' implementation
    # From Lalush and Wernick;
    # f^\hat <- (f^\hat / |\sum h|) * \sum h * (g_j / g)          ... (*)
    # where g = \sum (h f^\hat)                                   ... (**)
    #
    # self.f is the current estimate f^\hat
    # The following g from (**) is equivalent to g = \sum (h f^\hat)
    '''
    norm = True
    center = False
    fdtype = floatType
    f = np.ones((p,p), fdtype)
    
    for i in range(0, iterations):
        print("Iteration:", i)

#            print("Subset:", j)
        muFinite = len(mValues)
        muFinite = 257*128
        fL = f - BL
        fU = BU - f
        
        L = projector(BL, p, fdtype, mValues)
        U = projector(BU, p, fdtype, mValues)
        gL = projector(fL, p, fdtype, mValues)
        gU = projector(fU, p, fdtype, mValues)
#        L = projector(BL, p, fdtype)
#        U = projector(BU, p, fdtype)
#        gL = projector(fL, p, fdtype)
#        gU = projector(fU, p, fdtype)
    
        # form parenthesised term (g_j / g) from (*)
        rL = g_j - L
        rU = U - g_j
        for m in mValues:
            rL[m,:] /= gL[m,:]
            rU[m,:] /= gU[m,:]
    
        # backproject to form \sum h * (g_j / g)
        g_rL = backprojector(rL, p, norm, center, mValues=mValues) / muFinite
        g_rU = backprojector(rU, p, norm, center, mValues=mValues) / muFinite
    
        # Combine the upper/lower bounds and complex parts
        IL = fL*g_rL
        IU = fU*g_rU
        
        #combine to get f
        f = (IL*BU + IU*BL) / (IL + IU)
        
        if smoothReconMode > 0 and i % smoothIncrement == 0 and i > 0: #smooth to stem growth of noise
            fCenter = fftpack.ifftshift(f) #avoid padding issues with some smoothing algorithms by ensuring image is centered
            fReal = np.real(fCenter)
            fImag = np.imag(fCenter)
            if smoothReconMode == 1:
                print("Smooth TV")
                h = 0.5
                if i > smoothMaxIteration:
                    h /= 2.0
                fReal = denoise_tv_chambolle(fReal, h, multichannel=False)
                fImag = denoise_tv_chambolle(fImag, h, multichannel=False)
            elif smoothReconMode == 2:
                h = 1.0
                '''
                NLM Smoothing Notes:
                A higher h results in a smoother image, at the expense of blurring features. 
                For a Gaussian noise of standard deviation sigma, a rule of thumb is to choose the value of h to be sigma of slightly less.
                The image is padded using the reflect mode of skimage.util.pad before denoising.
                '''
                if i > smoothMidIteration:
                    h /= 2.0
                elif i > smoothMaxIteration:
                    h /= 4.0
                print("Smooth NL h:",h)
                fReal = denoise_nl_means(fReal, patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                fImag = denoise_nl_means(fImag, patch_size=5, patch_distance=11, h=h, multichannel=False, fast_mode=True).astype(fdtype)
            elif smoothReconMode == 3:
                print("Smooth Median")
                fReal = ndimage.median_filter(fReal, 3).astype(fdtype)
                fImag = ndimage.median_filter(fImag, 3).astype(fdtype)
            elif smoothReconMode == 4:
                print("Smooth Bilateral")
                fReal = denoise_bilateral(fReal, sigma_color=0.15, sigma_spatial=7, multichannel=False).astype(fdtype)
                fImag = denoise_bilateral(fImag, sigma_color=0.15, sigma_spatial=7, multichannel=False).astype(fdtype)
                
            f = fftpack.fftshift(fReal +1j*fImag)
        
    return f

#-------------------------------
#reconstruct test using MLEM   
start = time.time() #time generation
#recon = abmlem_expand(iterations, p, drtSpace, subsetsMValues, finite.frt_complex, finite.ifrt)
recon = abmlem_expand_complex(iterations, p, drtSpace, mValues, finite.frt_complex, finite.ifrt_complex)
#recon = fftpack.ifftshift(recon)
recon = np.abs(recon)
print("Done")
end = time.time()
elapsed = end - start
print("OSEM Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

mse = imageio.immse(image, recon)
ssim = imageio.imssim(image.astype(float), recon.astype(float))
psnr = imageio.impsnr(image, recon)
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)
diff = image - recon

#plot


pp = PdfPages('chaotic_abmlem_run.pdf')

fig, ax = plt.subplots(figsize=(24, 9))

plt.subplot(131)
#rax = plt.imshow(lena, interpolation='nearest', cmap='gray')
rax = plt.imshow(image, cmap='gray')
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Image')
plt.subplot(132)
#rax2 = plt.imshow(recon, interpolation='nearest', cmap='gray')
rax2 = plt.imshow(recon, cmap='gray')
rcbar2 = plt.colorbar(rax2, cmap='gray')
#plt.title('Reconstruction\n(i='+str(iterations)+', K:'+str(K)+', mu:'+str(mu)+', SNR='+str(SNR)+', Time='+'{:3.2f}'.format(elapsed)+')')
#plt.title('Reconstruction\n(SNR='+str(SNR)+', Time='+'{:3.2f}'.format(elapsed)+'s)')
plt.title('Reconstruction')
plt.subplot(133)
#rax3 = plt.imshow(diff, interpolation='nearest', cmap='gray', vmin=-24, vmax=24)
rax3 = plt.imshow(diff, cmap='gray')
rcbar3 = plt.colorbar(rax3, cmap='gray')
#plt.title('Reconstruction Errors\n(RMSE='+'{:3.2f}'.format(math.sqrt(mse))+')')
plt.title('Reconstruction Errors')
pp.savefig()

plt.tight_layout()

if plotSampling:
    #plot slices responsible for reconstruction    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    
    plt.gray()
    plt.tight_layout()
    
    maxLines = 0
    maxLines += len(lines)
    
    ax[0].imshow(powSpectGrid)
    ax[1].imshow(powSpectGrid)
    
    color=iter(cm.jet(np.linspace(0,1,maxLines+1)))
    fareyImage = np.zeros_like(powSpectGrid)
    for i, line in enumerate(lines):
        u, v = line
        c=next(color)
        ax[0].plot(u, v, '.', c=c)
        ax[1].plot(u, v, '.r',markersize=1)
        fareyImage[u,v] = 255
        if i == maxLines:
            break
    
    ax[0].set_title('Sampling (colour per line) for prime size:'+str(p))
    ax[1].set_title('Sampling (same colour per line) for prime size:'+str(p))
    #ax[0].set_xlim([0,M])
    #ax[0].set_ylim([0,M])
    #ax[1].set_xlim([0,M])
    #ax[1].set_ylim([0,M])
    plt.figure(10)
    plt.imshow(fareyImage)

plt.show()

print("Complete")
