# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:33:09 2018

@author: matthew
"""


from __future__ import division

#import os
#import sys
#file_dir = os.path.dirname(__file__)
#sys.path.append(file_dir)

import numpy as np
import scipy.fftpack as fftpack
#from matlab_tictoc import tic, toc
import param_class
import cs_utils_original as cs_utils
import pyfftw

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

def fnlCg(x0, params):

    maxlsiter = params.lnIterLim
    gradToll = params.gradToll
    alpha = params.lineSearchAlpha   
    beta = params.lineSearchBeta
    t0 = params.lineSearchT0
    k = 0
    t = 1
    
    x=x0
    g0 = wGradient(x,params)
    dx = param_class.JoinedWavelet(-g0.coeffs, g0.slices)
    while(1):
    
    # backtracking line-search
    
    	# pre-calculate values, such that it would be cheap to compute the objective
    	# many times for efficient line-search
        FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx = preobjective(x, dx, params)
        f0 = objective(FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx, x, dx, 0, params)
        t = t0
        f1  =  objective(FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx, x, dx, t, params)
    	
        lsiter = 0
        
        while ((f1 > f0 - alpha*t*np.abs(np.reshape(g0.coeffs,-1).conj().transpose()*np.reshape(dx.coeffs,-1)))).all() & (lsiter<maxlsiter):
        	lsiter = lsiter + 1
        	t = t * beta
        	f1  =  objective(FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx, x, dx, t, params)
            
        	if lsiter == maxlsiter:
        		print('Reached max line search,.... not so good... might have a bug in operators. exiting... ');
        #		return 1
        
        	# control the number of line searches by adapting the initial step search
        if lsiter > 2:
        	t0 = t0 * beta
        	
        if lsiter<1:
        	t0 = t0 / beta
        
        x = param_class.JoinedWavelet(x.coeffs + t*dx.coeffs, x.slices);
            #conjugate gradient calculation
            
        g1 = wGradient(x,params)
        bk = (np.matmul(np.reshape(g1.coeffs,-1).conj().transpose(),np.reshape(g1.coeffs,-1))
            /(np.matmul(np.reshape(g0.coeffs,-1).conj().transpose(),np.reshape(g0.coeffs,-1))
            +np.finfo(np.float32).eps))
        g0 = g1
        dx =  param_class.JoinedWavelet(- g1.coeffs + bk * dx.coeffs, dx.slices)
        k = k + 1
        	
        #TODO: need to "think" of a "better" stopping criteria ;-)
        if (k > params.Itnlim) | (np.linalg.norm(np.reshape(dx.coeffs,-1)) < gradToll):
            break
        
    return x


def preobjective(x, dx, params):
    #[FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx]
    # precalculates transforms to make line search cheap
    '''
    Should work now
    '''
    
    mx = cs_utils.idw2(x)
    mdx = cs_utils.idw2(dx)
    
    FTXFMtx = cs_utils.fft2u(mx, params)
    #FTXFMtx = FTXFMtx / np.abs(FTXFMtx).max()
    
    FTXFMtdx = cs_utils.fft2u(mdx, params)
    #FTXFMtdx = FTXFMtdx / np.abs(FTXFMtdx).max()
    
    if params.TVWeight > 0:
        DXFMtx = cs_utils.fd(mx)
        DXFMtdx = cs_utils.fd(mdx)
    else:
        DXFMtx = 0
        DXFMtdx = 0
    
    return FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx

def objective(FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx, x, dx, t, params):
    '''
    Should work now
    '''
    #calculated the objective function
    #[res, obj, RMS]
    p = params.pNorm
    
    obj = FTXFMtx + t*FTXFMtdx - params.data
    #obj = np.matmul(np.reshape(obj,-1).conj().transpose(),np.reshape(obj, -1))
    obj = np.linalg.norm(obj)**2
    
    if params.wavWeight > 0:
       w = np.reshape(x.coeffs,-1) + t*np.reshape(dx.coeffs,-1)
       XFM = (w*np.conj(w)+params.l1Smooth)**(p/2)
    else:
        XFM = 0;
    
    if params.TVWeight > 0:
       w = np.reshape(DXFMtx,-1) + t*np.reshape(DXFMtdx,-1)      
       TV = (w*np.conj(w)+params.l1Smooth)**(p/2)
    else:
        TV = 0
    
    XFM = np.sum(XFM*params.wavWeight)
    TV = np.sum(TV*params.TVWeight)
    #RMS = np.sqrt(obj/np.sum(np.abs(params.data[:])>0))
    res = params.consistencyWeight * obj  + XFM + TV
    print('obj ' + str(np.abs(obj)) + ' xfm ' + str(np.abs(XFM)) + ' tv ' + str(np.abs(TV)))
    
    return res

def wGradient(x,params):
    '''
    Return gradient of full objective function
    Should work now
    '''
    #grad
    gradXFM = 0
    
    gradObj = gOBJ(x, params)
    if params.wavWeight > 0:
        gradXFM = gXFM(x, params)
    if params.TVWeight > 0:
        gradTV = gTV(x, params)
    else:
        gradTV = 0
    
    
    grad = gradObj.coeffs +  params.wavWeight*gradXFM.coeffs + params.TVWeight*gradTV.coeffs
    return param_class.JoinedWavelet(grad, gradObj.slices)

def gOBJ(x,params):
    # computes the gradient of the data consistency
    '''
    Compute gradient of data consistency term ||F_u x - m||_2^2
    99% sure this works now
    '''
    image_space = cs_utils.idw2(x)
    fourier_space = cs_utils.fft2u(image_space, params)
    fourier_space = fourier_space/np.abs(fourier_space).max()
    fourier_diff = fourier_space - params.data
    image_space_diff = cs_utils.ifft2u(fourier_diff)
    wavelet_diff = cs_utils.dw2(image_space_diff)
    
    gradObjWavelet = param_class.JoinedWavelet(2*wavelet_diff.coeffs, wavelet_diff.slices)
    
    return gradObjWavelet
    
def gXFM(joined_wavelet,params):
    ''' 
    Compute gradient of sparsifying term
    99% sure this works now
    '''
    # compute gradient of the L1 transform operator
    x = joined_wavelet.coeffs
    p = params.pNorm
    
    grad = p*x / np.sqrt(x * np.conj(x) + params.l1Smooth)
    return param_class.JoinedWavelet(grad, joined_wavelet.slices)

def gTV(x,params):
# compute gradient of TV operator

    p = params.pNorm
    
    Dx = cs_utils.fd(cs_utils.idw2(x))
    
    G = p*Dx*(Dx*np.conj(Dx) + params.l1Smooth)**(p/2-1)
    grad = cs_utils.dw2((cs_utils.fd_adj(G)))
    return grad
