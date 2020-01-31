# -*- coding: utf-8 -*-
"""
Medical Imaging module featuring image processing etc.

Modules return Nifti images usually. The numpy array can be accessed by get_data() and the transform by get_affine()

Created on Sun Sep 20 11:33:03 2015

@author: shakes
"""
import scipy
import numpy as np
import nibabel as nib #nifti file support
from skimage import exposure #rescale intensities
from skimage import restoration #smoothing
from nilearn import image #plots/overlays

#smoothing
def smoothTV(img, weight=0.05):
    '''
    Denoise the image by smoothing using the Total Variation (TV) algorithm.
    Returns Nifti image
    '''
    result = restoration.denoise_tv_chambolle(img.get_data(), weight=weight, multichannel=False)
    
    return nib.Nifti1Image(result, img.get_affine())
    
def smoothBilateral(img, sigma_range=0.05, sigma_spatial=15):
    '''
    Denoise the image by smoothing using the Bilateral algorithm.
    Returns Nifti image
    '''
    result = restoration.denoise_bilateral(img.get_data(), sigma_range=sigma_range, sigma_spatial=sigma_spatial)
    
    return nib.Nifti1Image(result, img.get_affine())
    
def smoothNLMeans(img, patch_size=6, patch_distance=12, h=0.03):
    '''
    Denoise the image by smoothing using the non-local means algorithm.
    Returns Nifti image
    '''
    #result = denoise_nl_means(slice, patch_size=parameter[0], patch_distance=parameter[1], h=parameter[2], multichannel=False, fast_mode=True)
    result = restoration.nl_means_denoising(img.get_data(), patch_size=patch_size, patch_distance=patch_distance, h=h, multichannel=False, fast_mode=True) #11.3
    
    return nib.Nifti1Image(result, img.get_affine())

#intensities
def rescaleIntensities(img, out_range='dtype', percentile=True):
    '''
    Rescale intensities to get better contrast. Percentile correction is optional.
    Returns Nifti image
    '''
    if percentile:
        p2, p98 = np.percentile(img.get_data(), (2, 98))
        print "Image Intensity Percentiles:", p2, p98
        data_rescale = exposure.rescale_intensity(img.get_data(), in_range=(p2, p98), out_range=out_range)
    else:
        data_rescale = exposure.rescale_intensity(img.get_data(), out_range=out_range)
    return nib.Nifti1Image(data_rescale, img.get_affine())
    
def histogramEqualize(img):
    '''
    Apply histogram equalisation to get better contrast.
    Returns Nifti image
    '''
    data_rescale = exposure.equalize_hist(img.get_data())
    return nib.Nifti1Image(data_rescale, img.get_affine())
    
#features
def gradientMagnitude(img, sigma=0.05):
    '''
    Compute the gradient magnitude of the image.
    Returns Nifti image
    '''
    result = scipy.ndimage.gaussian_gradient_magnitude(img.get_data(), sigma)    
    return nib.Nifti1Image(result, img.get_affine())

#transform    
def applyOrientation(img, interpolation='continuous'):
    '''
    Same as resampleToMNISpace()
    '''
    return resampleToMNISpace(img, interpolation)
    
def resampleToMNISpace(img, interpolation='continuous'):
    '''
    Resample image by applying the orientation matrix (and origin) so that the result has an orientation of Identity (and zero origin).
    Returns Nifti image
    '''
    target_affine_3x3 = np.eye(3) #leads to re-estimation of bounding box
#    target_affine_4x4 = np.eye(4) #Uses affine anchor and estimates bounding box size
    return image.resample_img(img, target_affine=target_affine_3x3, interpolation=interpolation)

#convenience functions
def getPreprocessedImage(filename, out_range='dtype', preserveOrientation=False):
    '''
    Load, orient, preprocess 3D
    Preprocessing includes 98th percentile exclusion and rescale intensities to 0-1
    Returns Nifti image
    '''
    #load image file
    img = nib.load(filename)
    if not preserveOrientation:
        print "Apply Orientation"
        img = applyOrientation(img) #numpy array
    print "Rescaling Intensities"
    img = rescaleIntensities(img, out_range=(0, 1), percentile=False) #algorithms expect 0 to 1 values
    
    return img

def getPreprocessedSlice(filename, axis='x', out_range='dtype', flip=True):
    '''
    Load, orient, preprocess 3D image and extract middle slice for 2D analysis
    Preprocessing includes 98th percentile exclusion and rescale intensities to 0-1
    Returns Nifti image
    '''
    img = getPreprocessedImage(filename, out_range='dtype')
    
    image = img.get_data() #numpy array
    if axis == 'x':
        slice = image[image.shape[0]/2,:,:] #sagittal view, slice middle of x-z plane
    elif axis == 'y':
        slice = image[:,image.shape[1]/2,:] #coronal view, slice middle of x-z plane
    else:
        slice = image[:,:,image.shape[2]/2] #axial view, slice middle of x-z plane
        
    if flip:
        slice = np.fliplr(slice).T #flipped x-axis when reading
    print "Slice shape:", slice.shape
    print "Slice dtype:", slice.dtype
    
    return nib.Nifti1Image(slice, np.eye(4))
    
#collections
def averageImage(collection, preserveOrientation=False):
    '''
    Compute the average image of the collection of image (filenames) and return. Images are assume pre-registered
    '''
    images = []
    for file in collection:
        '''
        Every image is used to create detail and base layers
        '''
        img = getPreprocessedImage(file, (0, 1), preserveOrientation)
        
        images.append(img)
        
    result = np.zeros(images[0].shape)
    affine = images[0].get_affine()
    for img in images:
        result += img.get_data()
    result /= len(collection)
        
    return nib.Nifti1Image(result, affine)
