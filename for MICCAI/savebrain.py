# -*- coding: utf-8 -*-
"""
Load Brain images and test ABfMLEM reconstruction from square fractal sampling

@author: marlon
"""
import scipy.fftpack as fftpack
import pyfftw
import loadNii
import numpy as np
import PIL
import matplotlib.pyplot as plt

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()
#-------------------------------
#load kspace data
images, num_cases = loadNii.load_data("slices_pad")

for k, im in enumerate(images):
    images[k,:,:] = (255 * im) // np.max(im)
images = np.abs(images)
images = images.astype(np.uint8)

img = PIL.Image.fromarray(images[1200,:,:], mode=None)

img.save("brain_1200.png")