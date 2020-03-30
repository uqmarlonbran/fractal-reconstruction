# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:40:33 2019

Loading nii images

@author: marlon
"""
import os
import numpy as np
import nibabel as nib
import pyfftw
import scipy.fftpack as fftpack

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

'''
Reads data from nii, assuming it is single images with one channel.
'''
def load_data(foldername, num=None, singlePres=False):
    input_names = []

    for path, subdirs, files in os.walk(foldername + '/'):
        for name in files:
            input_names.append(os.path.join(path, name))
    
    num_cases = len(input_names)
    
    first_case = np.abs(nib.load(input_names[0]).get_data())
    rows, cols = first_case.shape
    
    inputs = np.zeros([num_cases, rows, cols], np.complex)
    
    count = 0
    for input_name in input_names:
        inputs[count, :, :] = (nib.load(input_name).get_data())
        count += 1
    print("Loaded ", count, "training images")
    
    if singlePres:
        for k, im in enumerate(inputs):
            inputs[k,:,:] = (255 * im) / np.max(im)
        # inputs = np.abs(inputs)
    
    if num and num <= num_cases:
        inputs = inputs[num,:,:]
    
    return inputs, num_cases

'''
Reads data from nii, assumes multiple channels
'''
def load_data_channels(foldername):
    input_names = []

    for path, subdirs, files in os.walk(foldername + '/'):
        for name in files:
            input_names.append(os.path.join(path, name))
    
    num_cases = len(input_names)
    
    first_case = np.abs(nib.load(input_names[0]).get_data())
    channels, rows, cols = first_case.shape
    
    inputs = np.zeros([num_cases, channels, rows, cols], np.complex)
    
    count = 0
    for input_name in input_names:
        inputs[count, :, :, :] = fftpack.ifftshift(nib.load(input_name).get_data())
        count += 1
    print("Loaded ", count, "training images")
    return inputs, num_cases