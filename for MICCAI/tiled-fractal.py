# -*- coding: utf-8 -*-
"""
Produce turbulent artefacts in k-space data.

This is a modified version of the script batch_artefacts.py created 
by uqscha22.

Created on Wed Jul 10 2019

@author: uqjwhi35
"""

# Load module for accessing files
import filenames
import sys
sys.path.insert(0, './fareyFunctions')
# Load modules for arrays and nifti file support
import numpy as np
import nibabel as nib
import finite
import scipy.fftpack as fftpack
import pyfftw
from fareyFractal import farey_fractal
import math
import os
# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()
import matplotlib.pyplot as plt

# Directory creator
def create_directory(dirName):
    try:
        # Create target directory
        os.mkdir(dirName[0:-1])
        print("Directory " , dirName[0:-1], " Created.")
    except FileExistsError:
        print("Directory " , dirName[0:-1], " already exists.")

###########
# FRACTAL #
###########
knees = "knees/"
N = 320    # Dimension of fractal 
K = 2.8   # Reduction factor
outpathFractal = knees + "fractal/"   # Outpath for the sampling fractal

# Create the directories required
create_directory(outpathFractal)

# Setup circular fractal
lines, angles, \
    mValues, fractalShakes, \
    oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', 
                                        twoQuads=True)
mu = len(lines)
print("Number of finite lines:", mu)
print("Number of finite points:", mu*(N))
print("Lines proportion:", mu/float(N))
fractalShakes = np.array(fftpack.fftshift(fractalShakes))

# Tile center region further
radius = N/8
centerX = N/2
centerY = N/2
count = 0
fractalTiled = np.array(fractalShakes)
for i, row in enumerate(fractalTiled):
    for j, col in enumerate(row):
        distance = math.sqrt( (i-float(centerX))**2 + (j-float(centerY))**2)
        if distance < radius:
            if not fractalTiled[i, j] > 0: #already selected
                count += 1
                fractalTiled[i, j] = 1
#fractalTiled = fftpack.fftshift(fractalTiled)
totalSamples = mu*(N)+count+1
actualR = float(totalSamples/N**2)
print("Number of total sampled points with centre tiling:", totalSamples)
print("Actual Reduction factor with centre tiling:", actualR)

fractalimg = nib.Nifti1Image(fractalTiled, np.eye(4))
outname = outpathFractal + "fractalTiledR2.nii.gz"
fractalimg.to_filename(outname)