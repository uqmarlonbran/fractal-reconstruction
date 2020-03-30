# -*- coding: utf-8 -*-
"""
Read Nifti 3D volumes via NiBabel and extract slices from 3D volume with 
multiple channels then save them as nifti files.

This is a modified version of the script batch_extract_slices.py created 
by uqscha22 and uqjwhi35.


@author: uqmbran
"""

# Load module for accessing files
import filenames

# Load modules for arrays and nifti file support
import numpy as np
import nibabel as nib

knees = "knees/"

path = knees + "images/"            # Path containing 3D volumes
kPath = knees + "knee_kspace/"        # Subdirectory containing k-space data
outpath = knees + "knee_kspace_slices/"         # Outpath for sliced images
outputPrefix = "case_"

caseIndex = 0    # Index of case number in filenames
sliceView = 0    # Which axis to take the slices along

sliceNum = 256   # Number of slices in each 3D volume

# Initiaise the list of offsets
offsets = []
for i in range(0, sliceNum):
    offsets.append(i)


kList = filenames.getSortedFileList(kPath, "*.nii.gz")


case = 0
for kName in kList:
    # Load nifti file
    k = nib.load(kPath+kName)
    print("Loaded", kName)
    
    # Get the numpy array version of the image
    data = k.get_data() #numpy array without orientation
    print("K Space shape:", data.shape)
    
    # Initialise the slice count
    count = 0
    print("Completed {0}/{1} slices".format(count, sliceNum), 
          end='\n', flush=True)
    
    # Extract each slice
    for offset in offsets:
    
        # Extract a slice from 3D volume to save. Note that the data 
        # is multi-channel, and thus each slice is also multi-channel. 
        # These channels constitute the fourth dimension of the data array
        if sliceView == 1: 
            kspace_slice = data[:,offset,:,:]
        elif sliceView == 0:
            kspace_slice = data[offset,:,:,:] 
        else:
            kspace_slice = data[:,:,offset,:]
     
        kspace_slice = np.fliplr(np.flipud(kspace_slice)) # Flipped x-axis and y-axis when reading
    
        # Save slice
        slice = nib.Nifti1Image(kspace_slice, np.eye(4))
        outname = (outpath + outputPrefix + str(case).zfill(3) + 
                   "_slice_" + str(count) + ".nii.gz")
        slice.to_filename(outname)
        count += 1
    
        print("Completed {0}/{1} slices".format(count, sliceNum),
             end='\n', flush=True)
    case += 1

print()

print("Slicing Complete")