# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:39:04 2020

@author: marlo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import scipy.io as io

# Number of samples
N = 160

# Types of recon schemes
NLCG = 0
GAN = 1
SIRT = 2

# Index for data array
OneD = 0
TwoD = 1
pfrac = 2
dfrac = 3

# Index for csv array
csv_oneD = 0
csv_twoD = 2
csv_pfrac = 4
csv_dfrac = 6

# Names of .mat


# Names of csv
csvs = ["R2.csv", "R4.csv", "R8.csv"]

# Score files GAN
GANPath = "GAN/"

# Score files NLCG
NLCGPath = "NLCG/"

# Score files SIRT
SIRTPath = "FFTSIRT/"

# Path to store boxplots
boxplotPath = "boxplots/"

reductionFactors = [2, 4, 8]

# Score arrays NLCG
matPathNLCG = NLCGPath + "MAT/"
# matsNLCG = ["NLCG_1D", "NLCG_2D", "NLCG_RAND", "NLCG_DET"]
matsNLCG = ["NLCG_1D", "NLCG_2D", "NLCG_RAND"]

# Score arrays SIRT
matPathSIRT = SIRTPath + "MAT/"
# matsSIRT = ["FFTSIRT_RAND_", "FFTSIRT_DET_"]
matsSIRT = ["FFTSIRT_RAND_"]

# Arrays to store everything
# psnr = np.zeros((3, len(reductionFactors), N, 4), dtype=np.float)
psnr = np.zeros((3, len(reductionFactors), N, len(matsNLCG)), dtype=np.float)
ssim = np.zeros_like(psnr)

# Get SIRT results for random fractal
for r in range(0, 3):
    string = str(reductionFactors[r]) + "_IMAGES.mat"
    for i, mat in enumerate(matsSIRT):
        dict_mat = io.loadmat(matPathSIRT + mat + string)
        psnr[SIRT, r, :, i] = np.squeeze(np.array(dict_mat.get('psnr')))
        ssim[SIRT, r, :, i] = np.squeeze(np.array(dict_mat.get('ssim')))

# Get NLCG results
for r in range(0, 3):
    string = str(reductionFactors[r]) + "_IMAGES.mat"
    for i, mat in enumerate(matsNLCG):
        dict_mat = io.loadmat(matPathNLCG + mat + string)
        psnr[NLCG, r, :, i] = np.squeeze(np.array(dict_mat.get('psnr')))
        ssim[NLCG, r, :, i] = np.squeeze(np.array(dict_mat.get('ssim')))

# Score arrays GAN
for r, c in enumerate(csvs):
    with open(GANPath + c) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            # Load data into the arrays row 162 is the average
            if line_count >= 2 and line_count != 162:
                psnr[GAN, r, line_count-2, OneD] = row[csv_oneD]
                ssim[GAN, r, line_count-2, OneD] = row[csv_oneD + 1]
                psnr[GAN, r, line_count-2, TwoD] = row[csv_twoD]
                ssim[GAN, r, line_count-2, TwoD] = row[csv_twoD + 1]
                psnr[GAN, r, line_count-2, pfrac] = row[csv_pfrac]
                ssim[GAN, r, line_count-2, pfrac] = row[csv_pfrac + 1]
                # psnr[GAN, r, line_count-2, dfrac] = row[csv_dfrac]
                # ssim[GAN, r, line_count-2, dfrac] = row[csv_dfrac + 1]
            line_count += 1
            
# Create Categories
# SIRTStrings = ['p.frac', 'd.frac']
SIRTStrings = ['p.frac']
# categoryStrings = ['Cart1D', 'Cart2D', 'p.frac', 'd.frac'] 
categoryStrings = ['1D', '2D', 'p.frac']   
# colors = ['steelblue', 'chocolate', 'forestgreen', 'firebrick']
colors = ['steelblue', 'chocolate', 'forestgreen']

# Make boxplots pretty
patch_artist = True
meanline = True
showmeans=True
medianprops=dict(color="black",linestyle='-',linewidth=6.0)
meanprops=dict(color="gold",linewidth=3.0)
widths = 0.6
font = {'size'   : 55,
        'weight' : 'bold'}
# 'weight' : 'bold',
whiskerprops = dict(linestyle='-',linewidth=6.0, color='black')
boxprops = dict(linestyle='-',linewidth=6.0, color='black')
capprops = dict(linestyle='-',linewidth=6.0, color='black')
plt.rc('font', **font)
rotation=90
for i, R in enumerate(reductionFactors):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 16), sharex='col', sharey='row', constrained_layout=True)
    
    # fig.suptitle("Reduction Factor: " + str(R))
    
    axs[0, 0].set_ylabel('PSNR (dB)')
    box = axs[0, 0].boxplot(psnr[SIRT, i, :, 0:len(SIRTStrings)], labels=SIRTStrings, widths=0.3, patch_artist=patch_artist, meanline=meanline, capprops=capprops, boxprops=boxprops, showmeans=showmeans, medianprops=medianprops, meanprops=meanprops, whiskerprops=whiskerprops)
    for f, patch, color in zip(box['fliers'], box['boxes'], colors[-len(SIRTStrings):]):
        patch.set_facecolor(color)
        f.set_markerfacecolor('black')
        f.set_marker("d")
        f.set_markersize(6)
    axs[0, 0].set_title('FFT-fSIRT')
    axs[0, 0].xaxis.set_tick_params(rotation=rotation)
    
    box = axs[0, 1].boxplot(psnr[NLCG, i, :, :], labels=categoryStrings, widths=widths, patch_artist=patch_artist, meanline=meanline, capprops=capprops, boxprops=boxprops, showmeans=showmeans, medianprops=medianprops, meanprops=meanprops, whiskerprops=whiskerprops)
    for f, patch, color in zip(box['fliers'], box['boxes'], colors):
        patch.set_facecolor(color)
        f.set_markerfacecolor('black')
        f.set_marker("d")
        f.set_markersize(6)
    axs[0, 1].set_title('NLCG')
    axs[0, 1].xaxis.set_tick_params(rotation=rotation)
    
    box = axs[0, 2].boxplot(psnr[GAN, i, :, :], labels=categoryStrings, widths=widths, patch_artist=patch_artist, meanline=meanline, capprops=capprops, boxprops=boxprops, showmeans=showmeans, medianprops=medianprops, meanprops=meanprops, whiskerprops=whiskerprops)
    for f, patch, color in zip(box['fliers'], box['boxes'], colors):
        patch.set_facecolor(color)
        f.set_markerfacecolor('black')
        f.set_marker("d")
        f.set_markersize(6)
    axs[0, 2].set_title('GAN-CS')
    axs[0, 2].xaxis.set_tick_params(rotation=rotation)
    
    axs[1, 0].set_ylabel('SSIM')
    box = axs[1, 0].boxplot(ssim[SIRT, i, :, 0:len(SIRTStrings)], labels=SIRTStrings, widths=0.3, patch_artist=patch_artist, capprops=capprops, boxprops=boxprops, meanline=meanline, showmeans=showmeans, medianprops=medianprops, meanprops=meanprops, whiskerprops=whiskerprops)
    for f, patch, color in zip(box['fliers'], box['boxes'], colors[-len(SIRTStrings):]):
        patch.set_facecolor(color)
        f.set_markerfacecolor('black')
        f.set_marker("d")
        f.set_markersize(6)
    axs[1, 0].xaxis.set_tick_params(rotation=rotation)
    
    box = axs[1, 1].boxplot(ssim[NLCG, i, :, :], labels=categoryStrings, widths=widths, patch_artist=patch_artist, meanline=meanline, capprops=capprops, boxprops=boxprops, showmeans=showmeans, medianprops=medianprops, meanprops=meanprops, whiskerprops=whiskerprops)
    for f, patch, color in zip(box['fliers'], box['boxes'], colors):
        patch.set_facecolor(color)
        f.set_markerfacecolor('black')
        f.set_marker("d")
        f.set_markersize(6)
    axs[1, 1].xaxis.set_tick_params(rotation=rotation)
    
    box = axs[1, 2].boxplot(ssim[GAN, i, :, :], labels=categoryStrings, widths=widths, patch_artist=patch_artist, meanline=meanline, capprops=capprops, boxprops=boxprops, showmeans=showmeans, medianprops=medianprops, meanprops=meanprops, whiskerprops=whiskerprops)
    for f, patch, color in zip(box['fliers'], box['boxes'], colors):
        patch.set_facecolor(color)
        f.set_markerfacecolor('black')
        f.set_marker("d")
        f.set_markersize(6)
    axs[1, 2].xaxis.set_tick_params(rotation=rotation)
    
    # axs[1, 0].set_aspect(2)
    # axs[0, 0].set_aspect(2)
    
    plt.savefig(boxplotPath + "r_" + str(R) + "_comp" + ".eps")
    fig.savefig(boxplotPath + str(R) + ".png")
    plt.close(fig)

