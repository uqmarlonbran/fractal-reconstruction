# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:53:18 2020

@author: marlo
"""
import imageio
from skimage import transform
import os
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Gets the important areas of the image
def trim(a, rotate=False, special=False):
    if special:
        a = np.flip(a)
        a = a[:, 0:256]
    else:
        a = a[30:226, 60:115]
    if rotate: a = np.fliplr(a.T)
    return a

# Get just the brain
def trim_og(a, rotate=False):
    
    a = a[30:226, 20:240]
    if rotate: a = np.fliplr(a.T)
    return a

# Get the directories for each of the different reconstructions
path_1D = '1D_Cart/'
path_2D = '2D_Cart/'
path_dfrac = 'd_frac/'
path_pfrac = 'p_frac/'

# Get paths for reduction factors
path_r2 = 'reduction_factor_2a.png'
path_r4 = 'reduction_factor_4a.png'
path_r8 = 'reduction_factor_8a.png'

# Create an array of required paths
path_stubs = [path_1D, path_2D, path_pfrac, path_dfrac]
rs = [path_r2, path_r4, path_r8]

# Format the images into the mega boi
for i, r in enumerate(rs):
    if i == 0:
        R = 2
    elif i == 1:
        R = 4
    else:
        R = 8
    first = True
    for i, path in enumerate(path_stubs):
        path_to_image = path + r
        image = imageio.imread(path_to_image)
        image = trim(image, rotate=False, special=True)
        image = image / np.max(image)
        image *= 255
        im = Image.fromarray((image).astype(np.uint8))
        im.save(path + "reduction_factor_" + str(R) + '.png')

