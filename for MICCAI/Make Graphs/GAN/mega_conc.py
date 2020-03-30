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
import csv
import scipy.io as io

# Gets the important areas of the image
def trim(a, rotate=False, special=False):
    if special:
        a = a[30:226, 60:116]
    else:
        a = a[30:226, 60:115]
    if rotate: a = np.fliplr(a.T)
    return a

# Get just the brain
def trim_og(a, rotate=False):
    
    a = a[30:226, 20:240]
    if rotate: a = np.fliplr(a.T)
    return a

reductionFactors = [2, 4, 8]

psnr = np.zeros((4, 3), dtype=float)
ssim = np.zeros_like(psnr)

# Score files
matPath = "MAT/"
mats = ["FFTSIRT_RAND_"]

# Score files
csvs = ["R2.csv", "R4.csv", "R8.csv"]
# Score arrays
for i, c in enumerate(csvs):
    with open(c) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Sample is 13th but csv has additional headers
        line = 14
        line_count = 0
        for row in csv_reader:
            if line_count == line:
                psnr[0, i] = row[0]
                ssim[0, i] = row[1]
                psnr[1, i] = row[2]
                ssim[1, i] = row[3]
                psnr[2, i] = row[4]
                ssim[2, i] = row[5]
                psnr[3, i] = row[6]
                ssim[3, i] = row[7]
            line_count += 1
            
# Get NLCG results
for r in range(0, 3):
    string = str(reductionFactors[r]) + "_IMAGES.mat"
    for i, mat in enumerate(mats):
        dict_mat = io.loadmat(matPath + mat + string)
        buffPSNR = np.squeeze(np.array(dict_mat.get('psnr')))
        buffSSIM = np.squeeze(np.array(dict_mat.get('ssim')))
        psnr[3, r] = buffPSNR[12]
        ssim[3, r] = buffSSIM[12]
# Get directory for clean image
path_clean = 'clean/sub-OAS30002_ses-d0653_run-01_T1w_00105.png'

# Get the directories for each of the different reconstructions
path_1D = '1D_Cart/'
path_2D = '2D_Cart/'
path_dfrac = 'd_frac/'
path_pfrac = 'p_frac/'
path_fftsirt = 'fftsirt/'

# Get paths for reduction factors
path_r2 = 'reduction_factor_2.png'
path_r4 = 'reduction_factor_4.png'
path_r8 = 'reduction_factor_8.png'

# Create an array of required paths
# path_stubs = [path_1D, path_2D, path_pfrac, path_dfrac]
path_stubs = [path_1D, path_2D, path_pfrac, path_fftsirt]
rs = [path_r2, path_r4, path_r8]

# Get the clean image nad put it into the big image
mega_conc = imageio.imread(path_clean)
mega_conc = mega_conc / np.max(mega_conc)
mega_conc = np.round(255 * mega_conc)
mega_conc = trim_og(mega_conc, rotate=True)

# Format the images into the mega boi
for r in rs:
    first = True
    for i, path in enumerate(path_stubs):
        path_to_image = path + r
        image = imageio.imread(path_to_image)
        image = trim(image, rotate=True)
        # image = image * 255 / np.max(image)
        if first:
            image_row = image
            first = False
        else:   
            image_row = np.r_[image_row, image]
    
    mega_conc = np.c_[mega_conc, image_row]






###
plt.gray()
fig = plt.figure(frameon=False)
# Use the width of Springer as a target
template_width_in = 12.2/2.54 - 0.05  # Springer LNCS textwidth converted to cm, with padding
h = template_width_in * mega_conc.shape[0] / mega_conc.shape[1]
dpi = mega_conc.shape[1] / template_width_in
fig.set_size_inches(template_width_in, h)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# Text overlay
font1 = {'family': 'serif',
        'color':  'cyan',
        'weight': 'normal',
        'horizontalalignment': 'left',
        'verticalalignment': 'top',
        'size': 6,
        }
font2 = {'family': 'serif',
        'color':  'cyan',
        'weight': 'normal',
        'horizontalalignment': 'right',
        'verticalalignment': 'top',
        'size': 6,
        }
x01 = 200
y01 = 0
dy1 = 55
ax.text(x01, y01+3, '1D', fontdict=font1)
ax.text(x01, y01+1*dy1, '2D', fontdict=font1)
ax.text(x01, y01+2*dy1, 'p.frac', fontdict=font1)
# ax.text(x01, y01+3*dy1, 'd.frac', fontdict=font1)
ax.text(x01, y01+3*dy1, 'FFT-fSIRT'+'\n'+'p.frac', fontdict=font1)

x02 = 385
dx2 = 196
y02 = 0
dy2 = 55
x2 = x02
for i in range(3):
    for j in range(4):
        text = f'{psnr[j,i]:.1f}' + f'\n{ssim[j,i]:.3f}'.replace('0.', '.')
        if j == 0:
            ax.text(x2, y02+j*dy2+3, text, fontdict=font2)
        else:
            ax.text(x2, y02+j*dy2, text, fontdict=font2)
    x2 += dx2+3


# Add crop outline
left, width = 0, .25
# bottom, height = .5273, .1992
bottom, height = .57, .25
right = left + width
top = bottom + height
p = patches.Rectangle(
    (left, bottom), width, height, 
    fill=False, transform=ax.transAxes, clip_on=False,
    edgecolor='cyan', linestyle=':', linewidth=.75,
    )
ax.add_patch(p)

ax.imshow(mega_conc, aspect='auto')
plt.savefig("test.eps", format='eps', dpi=dpi)
plt.savefig("test.png", format='png', dpi=dpi)
###