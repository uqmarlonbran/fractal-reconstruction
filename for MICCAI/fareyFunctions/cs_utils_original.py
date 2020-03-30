# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:04:58 2018

@author: matthew
"""
import numpy as np
import scipy.fftpack as fftpack
import pywt
import param_class
import pyfftw

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

def total_variation(image):
    pixel_difY = np.abs(image[1:, :] - image[:-1, :])
    pixel_difX = np.abs(image[:, 1:] - image[:, :-1])
    
    pixel_difY_sum = np.sum(pixel_difY)
    pixel_difX_sum = np.sum(pixel_difX)
    
    return pixel_difY_sum + pixel_difX_sum

def fd(image):
    diffX = np.diff(image, n=1, axis=0)
    diffX = np.pad(diffX, [(0,1),(0,0)],mode='constant')
    
    diffY = np.diff(image, n=1, axis=1)    
    diffY = np.pad(diffY, [(0,0),(0,1)],mode='constant')
    
    return np.stack((diffX,diffY))    

def fd_adj(fd):
    X = fd_dx(fd[0,:,:])
    Y = fd_dy(fd[1,:,:])
    return X + Y

def fd_dy(x):
    (maxX, maxY) = x.shape
    ind = np.arange(0, maxY-1)
    ind=np.insert(ind, 0, 0)
    res = x[:,ind] - x;
    res[:,0] = -x[:,0];
    res[:,-1] = x[:,-1];
    return res

def fd_dx(x):
    (maxX, maxY) = x.shape
    ind = np.arange(0, maxX-1)
    ind=np.insert(ind, 0, 0)
    res = x[ind,:] - x;
    res[0,:] = -x[0,:];
    res[-1,:] = x[-1,:];
    return res
    
def fft2u(image, params):
    sampling = params.FTMask
#    return 1/np.sqrt(image.size)*fftpack.fft2(image)*sampling
    #return fftpack.ifftshift(fftpack.fftshift(fftpack.fft2(image))*sampling)
#    return 1/np.sqrt(image.size)*fftpack.fftshift(fftpack.fft2(image))*sampling
    return 1/np.sqrt(image.size)*fftpack.fftshift(fftpack.fft2(fftpack.ifftshift(image)))*sampling

def ifft2u(kspace):
#    return np.sqrt(kspace.size)*fftpack.ifft2(kspace)
#    return fftpack.ifftshift((fftpack.ifft2((kspace))))
#    return np.sqrt(kspace.size)*fftpack.ifft2(fftpack.fftshift(kspace))
    return np.sqrt(kspace.size)*fftpack.ifftshift(fftpack.ifft2(fftpack.fftshift(kspace)))

def dw2(image, wavelet='db4', mode='periodization'):
    coeffs = pywt.wavedec2(image,wavelet=wavelet, level=4,mode=mode)
    joinedCoeffs,slices = pywt.coeffs_to_array(coeffs)
    return param_class.JoinedWavelet(joinedCoeffs, slices)

def idw2(joined_wavelet, wavelet='db4', mode='periodization'):
    splitCoeffs = joined_wavelet.splitCoeffs()
    return pywt.waverec2(splitCoeffs, wavelet=wavelet,mode=mode)
    
#def dw2_complex(image, wavelet='db4', mode='periodic'):
#    coeffsReal = pywt.wavedec2(image.real,wavelet=wavelet, mode=mode)
#    joinedCoeffsReal,slicesReal = pywt.coeffs_to_array(coeffsReal)
#    
#    coeffsImag = pywt.wavedec2(image.imag,wavelet=wavelet, mode=mode)
#    joinedCoeffsImag,slicesImag = pywt.coeffs_to_array(coeffsImag)
#
#    return param_class.ComplexWavelet(image, joinedCoeffsReal,slicesReal,joinedCoeffsImag,slicesImag)
#
#def idw2_complex(ComplexWavelet, slices, wavelet='db4', mode='periodic'):
#    coeffsReal = ComplexWavelet.real.coeffs
#    coeffsImag = ComplexWavelet.imag.coeffs
#    
#    slicesReal = ComplexWavelet.real.slices
#    slicesImag = ComplexWavelet.imag.slicse
#    
#    coeffsRealSplit = pywt.array_to_coeffs(coeffsReal, slicesReal, output_format='wavedec2')
#    imReal = pywt.waverec2(coeffsRealSplit, wavelet=wavelet, mode=mode)
#
#    coeffsImagSplit = pywt.array_to_coeffs(coeffsImag, slicesImag, output_format='wavedec2')
#    imImag = pywt.waverec2(coeffsImagSplit, wavelet=wavelet, mode=mode)
#
#    return imReal + 1j *imImag