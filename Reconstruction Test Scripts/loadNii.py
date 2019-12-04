# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:40:33 2019

Loading nii images

@author: marlon
"""
import os
import numpy as np
import nibabel as nib

'''
Reads data from nii, assuming it is single images.
'''
def load_data(foldername):
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
        inputs[count, :, :] = np.abs(nib.load(input_name).get_data())
        count += 1
    print("Loaded ", count, "training images")
    return inputs, num_cases