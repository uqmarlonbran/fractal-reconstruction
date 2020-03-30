"""
Generate undersampled and ground truth images from 3D k-space data

Created on Mon Jul 15 2019

@author: uqjwhi35
"""

# Extract slices
print("EXTRACTING SLICES")
import extract_slices

# Apply the fractal sampling pattern
print("APPLYING UNDERSAMPLING")
import undersample

# Reconstruct the images from the k-space data
print("RECONSTRUCTING IMAGES")
import reconstruct
