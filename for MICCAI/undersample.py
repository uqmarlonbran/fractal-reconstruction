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

N = 320    # Dimension of fractal 
K = 2.25   # Reduction factor

knees = "knees/"

path = knees + "knee_kspace_slices/"               # Path
outpathArtefactShakes = knees + "knee_kspace_slices_shakes_artefact/"   # Outpath for turbulent slices 
outpathArtefactTiled = knees + "knee_kspace_slices_tiled_artefact/"   # Outpath for turbulent slices 
outpathArtefactMarlon = knees + "knee_kspace_slices_marlon_artefact/"   # Outpath for turbulent slices
outpathFractal = knees + "fractal/"   # Outpath for the sampling fractal
outputPrefix = "case_"

# Create the directories required
create_directory(outpathArtefactShakes)
create_directory(outpathArtefactTiled)
create_directory(outpathArtefactMarlon)
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



lines, angles, \
    mValues, fractalMarlon, \
    R, oversampleFilter = farey_fractal(N, 12, centered=True)
fractalMarlon = np.array(fractalMarlon)
# Save an image of the fractal
fractalimg = nib.Nifti1Image(fractalShakes, np.eye(4))
outname = outpathFractal + "fractalShakes.nii.gz"
fractalimg.to_filename(outname)

fractalimg = nib.Nifti1Image(fractalMarlon, np.eye(4))
outname = outpathFractal + "fractalMarlon.nii.gz"
fractalimg.to_filename(outname)

fractalimg = nib.Nifti1Image(fractalTiled, np.eye(4))
outname = outpathFractal + "fractalTiled.nii.gz"
fractalimg.to_filename(outname)

# plt.figure(1)
# plt.imshow(np.abs(fractalShakes))
# plt.figure(2)
# plt.imshow(np.abs(fractalMarlon))
# plt.figure(3)
# plt.imshow(np.abs(fractalTiled))

#####################
# PROCESSING SLICES #
#####################

caseIndex = 0    # Index of case number in slice filenames

# Get the files containing the original slices
kList, caseList = filenames.getSortedFileListAndCases(path, 
                                                          caseIndex, 
                                                          "*.nii.gz", True)
kList, sliceList = filenames.getSortedFileListAndCases(path, 
                                                           caseIndex+1,
                                                           "*.nii.gz", True)

# Process each slice
count = 0
for kName, case, sliceIndex in zip(kList, caseList, sliceList):
    k = nib.load(kName)
    print("Loaded", kName)

    # Get the numpy array version of the image
    data = k.get_data() # numpy array without orientation
    channels, lx, ly = data.shape
    print("Image shape:", data.shape)
    
    artefactkSpaceShakes = np.zeros((channels, N, N), dtype = complex)
    artefactkSpaceMarlon = np.zeros((channels, N, N), dtype = complex)
    artefactkSpaceTiled = np.zeros((channels, N, N), dtype = complex)

    # Process each channel of the data individually
    for channel in range(0, channels):
        artefactkSpaceShakes[channel, :, :] = data[channel, :, :] * fractalShakes
        artefactkSpaceMarlon[channel, :, :] = data[channel, :, :] * fractalMarlon
        artefactkSpaceTiled[channel, :, :] = data[channel, :, :] * fractalTiled

    # Save the downsampled kspace SHAKES
    slice = nib.Nifti1Image(artefactkSpaceShakes, np.eye(4))
    outname = (outpathArtefactShakes + outputPrefix + str(case).zfill(3) + 
               "_slice_" + str(sliceIndex) + ".nii.gz")
    slice.to_filename(outname)
    
    # Save the downsampled kspace TILED
    slice = nib.Nifti1Image(artefactkSpaceTiled, np.eye(4))
    outname = (outpathArtefactTiled + outputPrefix + str(case).zfill(3) + 
               "_slice_" + str(sliceIndex) + ".nii.gz")
    slice.to_filename(outname)
    
    # Save the downsampled kspace MARLON
    slice = nib.Nifti1Image(artefactkSpaceMarlon, np.eye(4))
    outname = (outpathArtefactMarlon + outputPrefix + str(case).zfill(3) + 
               "_slice_" + str(sliceIndex) + ".nii.gz")
    slice.to_filename(outname)
    count += 1
      

    
print("Total", count, "processed")