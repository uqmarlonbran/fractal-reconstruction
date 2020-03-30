# -*- coding: utf-8 -*-
"""
Reconstruct images from multichanel k-space data.

This is a modified version of the script batch_artefacts.py created 
by uqscha22.

Created on Wed Jul 10 2019

@author: uqjwhi35
"""

# Load module for accessing files
import filenames

# Load modules for arrays and nifti file support
import numpy as np
import nibabel as nib
import scipy.fftpack as fftpack
import pyfftw
import os

# Directory creator
def create_directory(dirName):
    try:
        # Create target directory
        os.mkdir(dirName[0:-1])
        print("Directory " , dirName[0:-1], " Created.")
    except FileExistsError:
        print("Directory " , dirName[0:-1], " already exists.")


# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

knees = "knees/"

pathArtefactShakes = knees + "knee_kspace_slices_shakes_artefact/"   # Path for turbulent slices 
pathArtefactTiled = knees + "knee_kspace_slices_tiled_artefact/"   # Path for turbulent slices 
pathArtefactMarlon = knees + "knee_kspace_slices_marlon_artefact/"   # Path for turbulent slices

path = knees + "knee_kspace_slices/"                    # Path for original k-space slices
outpathArtefactShakes = knees + "knee_slices_shakes_artefact/"   # Outpath for turbulent slices 
outpathArtefactTiled = knees + "knee_slices_tiled_artefact/"   # Outpath for turbulent slices 
outpathArtefactMarlon = knees + "knee_slices_marlon_artefact/"   # Outpath for turbulent slices
                                            #     turbulent slices
outpathSlices = knees + "knee_slices/"            # Outpath for reconstructed 
                                            #     original slices

outputPrefix = "case_"

# Create the directories required
create_directory(outpathArtefactShakes)
create_directory(outpathArtefactTiled)
create_directory(outpathArtefactMarlon)
create_directory(outpathSlices)


# Structure containing the paths for the turbulent and original slices
dataPaths = [[path, outpathSlices], [pathArtefactTiled, outpathArtefactTiled], [pathArtefactShakes, outpathArtefactShakes], [pathArtefactMarlon, outpathArtefactMarlon]]
pathIndex = 0   # Index of the input path in each sub-list
outputIndex = 1 # Index of the outpath in each sub-list

complexOutput = 1 # Flag for if the final image can have complex values

# Reconstruct the images
for imageType in dataPaths:

    caseIndex = 0 # Index of case number in filenames

    # Get the files containing the slices
    kList, \
        caseList = filenames.getSortedFileListAndCases(imageType[pathIndex], 
                                                       caseIndex, 
                                                       "*.nii.gz", True)
    kList, \
        sliceList = filenames.getSortedFileListAndCases(imageType[pathIndex], 
                                                        caseIndex + 1, 
                                                        "*.nii.gz", True)

    count = 0 # Number of slices processed

    # Process each slice
    for kName, case, sliceIndex in zip(kList, caseList, sliceList):
        k = nib.load(kName)
        print("Loaded", kName)

        # Get the numpy array version of the image
        data = k.get_data() #numpy array without orientation
        channels, lx, ly = data.shape
        print("Image shape:", data.shape)
    
        # 2D FFT
        newImage = np.zeros((lx, ly), dtype = complex)

        # Combine the data from each channel (Note data is not centred)
        for channel in range(0, channels):
            newImageSlice = fftpack.ifft2(fftpack.ifftshift(data[channel, :, :]))
            newImage += (newImageSlice ** 2)

        newImage = np.sqrt(newImage)
        if not complexOutput:
            newImage = np.absolute(newImage)
        

        # Save the output image
        slice = nib.Nifti1Image(newImage, np.eye(4))
        outname = (imageType[outputIndex] + outputPrefix + 
                   str(case).zfill(3) + "_slice_" + str(sliceIndex) + 
                   ".nii.gz")
        slice.to_filename(outname)
        count += 1

 
    print("Total", count, "processed") 
