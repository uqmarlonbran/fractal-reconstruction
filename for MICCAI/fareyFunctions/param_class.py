# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:58:24 2018

@author: matthew
"""
import pywt

class nlcg_param:
    def __init__(self):
        self.lnIterLim = 150
        self.Itnlim = 8
        self.pNorm = 1
        self.FTMask = 0
        self.data = 0
        self.TVWeight = 0
        self.wavWeight = 0
        self.maxiter = 10
        self.gradToll = 1e-30
        self.l1Smooth = 1e-15
        self.lineSearchAlpha = 0.05
        self.lineSearchBeta = 0.6
        self.lineSearchT0 = 1
        self.consistencyWeight = 1

class JoinedWavelet:
    def __init__(self, coeffs, slices):
        self.coeffs = coeffs
        self.slices = slices
    def splitCoeffs(self):
        return pywt.array_to_coeffs(self.coeffs,self.slices,output_format='wavedec2')
        
class ComplexWavelet:
    def __init__(self, image, coeffsReal,slicesReal,coeffsImag,slicesImag):
        self.image = image
        self.real = JoinedWavelet(coeffsReal, slicesReal)
        self.imag = JoinedWavelet(coeffsImag, slicesImag)