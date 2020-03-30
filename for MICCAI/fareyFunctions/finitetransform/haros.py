# -*- coding: utf-8 -*-
"""
New class for creating the Farey-Haros sequence using Gaussian Integers

Created on Fri Oct 06 09:49:50 2017

@author: uqscha22
"""
from finitetransform.numbertheory import GaussianInteger

def haros(a, b):
    '''
    Convenience member for creating a Farey vector from a Farey fraction a/b
    '''
    return GaussianInteger(a, b)
    
def get_ab(angle):
    '''
    Return p, q tuple of the angle provided using module convention
    '''
    b = int(angle.imag)
    a = int(angle.real)
    
    return a, b
    
def projectionLength(angle, P, Q):
    '''
    Return the number of bins for projection at angle of a PxQ image.
    '''
    p, q = get_ab(angle)
    return (Q-1)*abs(p)+(P-1)*abs(q)+1 #no. of bins
    
def isKatzCriterion(P, Q, angles, K = 1):
    '''
    Return true if angle set meets Katz criterion for exact reconstruction of
    discrete arrays
    '''
    sumOfA = 0
    sumOfB = 0
    n = len(angles)
    for j in range(0, n):
        a, b = get_ab(angles[j])
        sumOfA += abs(a)
        sumOfB += abs(b)
        
#    if max(sumOfP, sumOfQ) > max(rows, cols):
    if sumOfA > K*P or sumOfB > K*Q:
        return True
    else:
        return False

class Haros:
    '''
    Class for the Farey-Haros vectors. It uses Gaussian integers and genrators to represent and create them
    
    Conventions used in theis class:
    Farey fraction a/b is represented as a vector (a, b) in (x, y) coordinates
    '''
    #static constants
    startVector = GaussianInteger(0, 1)
    endVector = GaussianInteger(1, 1)

    def __init__(self):
        #class variables
        self.vector = GaussianInteger(0, 0)
        self.generated = False
#        self.generatedFinite = False
#        self.compact = False
        self.vectors = []
#        self.finiteAngles = []
        
    def computeNext(self, n):
        '''
        Use the formula for Farey to directly generate the next member of the Farey sequence of order n. Generator version.
        
        If b is the next (unknown) term in the Farey sequence given the first term a and c the mediant term then
        a + b = c. Since c is in lowest terms, there is an integer k, so that kc = a + b.
        Thus, b = kc - a.
        But we want Farey sequence with imaginary part at most n, so k <= (n+a.imag)/c.imag
        '''
        nthVector = GaussianInteger(1, n) # 1/n
        angle1 = self.startVector # 0/1
        angle2 = nthVector
        
        yield self.startVector
        yield nthVector
        
        nextAngle = GaussianInteger(0, 0)
        while nextAngle != self.endVector: # 1/1
            k = int( (n+angle1.imag) / float(angle2.imag) )
            nextAngle = k*angle2 - angle1
    
            yield nextAngle
    #            print nextAngle
            angle1 = angle2
            angle2 = nextAngle
            
    def generate(self, n, octants=1):
        '''
        Generate all the Farey-Haros vectors up to given n.
        Octants is the number of octants to produce, 1 is the first octant, 2 is the first two octants, 4 is first two quadrants and > 4 is all quadrants
        '''
        for level in range(0,n):
            self.vectors = list(self.computeNext(n))
            
        if octants > 1:
            secondOctantVectors = []
            for nextAngle in self.vectors:
                if not nextAngle.imag == nextAngle.real:
                    nextOctantAngle = haros(nextAngle.imag, nextAngle.real) #mirror
                    secondOctantVectors.append(nextOctantAngle)
            self.vectors += secondOctantVectors #merge lists
            
        #use four-fold symmetry (by rotation in complex plane) to determine rest of the quadrants
        secondQuadrantVectors = []
        if octants > 2:
            for nextAngle in self.vectors:
                if not (nextAngle.imag == 0 or nextAngle.real == 0):
                    secondQuadrantVectors.append(1j*nextAngle)
                    
        thirdQuadrantVectors = []
        if octants > 4:
            for nextAngle in secondQuadrantVectors:
                if not (nextAngle.imag == 0 or nextAngle.real == 0):
                    thirdQuadrantVectors.append(1j*nextAngle)
                    
        forthQuadrantVectors = []
        if octants > 6:
            for nextAngle in thirdQuadrantVectors:
                if not (nextAngle.imag == 0 or nextAngle.real == 0):
                    forthQuadrantVectors.append(1j*nextAngle)
                    
        self.vectors += secondQuadrantVectors #merge lists
        self.vectors += thirdQuadrantVectors #merge lists
        self.vectors += forthQuadrantVectors #merge lists
           
        self.generated = True

def angleSet_Symmetric(P, Q, octant=0, binLengths=False, K = 1):
    '''
    Generate the minimal L1 angle set for the MT.
    Parameter K controls the redundancy, K = 1 is minimal.
    If octant is non-zero, full quadrant will be used. Octant schemes are as follows:
        If octant = -1, the opposing octant is also used.
        If octant = 0,1 (default), only use one octant.
        If octant = 2, octant will be mirrored from diagonal to form a quadrant.
        If octant = 4, 2 quadrants.
        If octant = 8, all quadrants.
    Function can also return bin lengths for each bin.
    '''
    maxPQ = max(P,Q)

    harosVectors = Haros()
    harosVectors.generate(maxPQ-1, 1) #always first octant because need to handle Katz later
    vectors = harosVectors.vectors
    sortedVectors = sorted(vectors, key=lambda x: x.real**2+x.imag**2) #sort by L2 magnitude
    
    index = 0
    angles = []
    binLengthList = []
    angles.append(sortedVectors[index])
    binLengthList.append(projectionLength(sortedVectors[index],P,Q))
    while not isKatzCriterion(P, Q, angles, K) and index < len(sortedVectors): # check Katz
        index += 1
        a, b = get_ab(sortedVectors[index]) # b = imag, a = real
        angles.append(sortedVectors[index])
        binLengthList.append(projectionLength(sortedVectors[index],P,Q))
    
        if octant == 0:
            continue
        
        #add octants
        if octant == -1:
            nextOctantAngle = haros(a, -b) #mirror from axis
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
        if octant > 0 and a != b:
            nextOctantAngle = haros(b, a) #swap to mirror from diagonal
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
        if octant > 1:
            nextOctantAngle = haros(a, -b) #mirror from axis
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if a != b: #dont replicate
                nextOctantAngle = haros(b, -a) #mirror from axis and swap to mirror from diagonal
                angles.append(nextOctantAngle)
                binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if binLengths:
        return angles, binLengthList
    return angles
    
def angleCoordinates(angle, b, N, center=False, modulus=False):
    '''
    Compute the 2D coordinates of each translate (in NxN DFT space) of farey angle of length b.
    b is the number of samples for each side of the DC
    N is assumed to be the larger space to embed lines
    Returns a list of u, v coordinate arrays [[u_0[...],v_0[...]], [u_1[...],v_1[...]], ...] for angle
    '''
    offset = 0
    if center:
#        offset = haros( int(N.real/2), int(N.imag/2) ) #integer division deliberate
        offset = haros( int(N.real/2), int(N.real/2) ) #integer division deliberate, assume mod N
#        print("Center:", offset)
    
    points = []
    #forward
    for translate in range(0, int(b)):
        multiple = angle*translate
        if modulus:
            multiple %= N
        multiple += offset
#        print "t:", translate, "multiple:", multiple
        points.append( multiple ) 
    #conjugate sym
    for translate in range(0, int(b)):
        multiple = 1j*1j*angle*translate
        if modulus:
            multiple %= N
        multiple += offset
#        print "t:", translate, "multiple:", multiple
        points.append( multiple ) 

    return points

def angleCoordinates2(angle, b, N, center=False, modulus=False):
    '''
    Compute the 2D coordinates of each translate (in NxN DFT space) of farey angle of length b.
    b is the number of samples for each side of the DC
    N is assumed to be the larger space to embed lines
    Returns a list of u, v coordinate arrays [[u_0[...],v_0[...]], [u_1[...],v_1[...]], ...] for angle
    '''
    offset = 0
    if center:
#        offset = haros( int(N.real/2), int(N.imag/2) ) #integer division deliberate
        offset = haros( int(N.real/2), int(N.real/2) ) #integer division deliberate, assume mod N
#        print("Center:", offset)
    
    points = []
    #forward
    for translate in range(0, int(b)):
        multiple = angle*translate
        if modulus:
            multiple %= N
        multiple += offset
        if modulus:
            multiple %= N
#        print "t:", translate, "multiple:", multiple
        points.append( multiple ) 

    return points
