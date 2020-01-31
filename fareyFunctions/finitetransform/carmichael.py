# -*- coding: utf-8 -*-
"""
Carmichael number transforms module
"""
import numpy as np
import numbertheory as nt

MODULUS = np.uint64(np.iinfo(np.uint32).max + 1) #2^M
MAXLENGTH = np.uint64(MODULUS/np.uint64(4)) #2^(M-2)
PRIMITIVEROOT = np.uint64(5)
NTT_FORWARD = -1
NTT_INVERSE = 1

def integer(value):
    '''
    Declare integer of appropriate type to use with Carmichael transforms.
    Normally this is a 64-bit unsigned integer that numpy supports.
    '''
    return np.uint64(value)
    
def signed_integer(value):
    '''
    Declare signed integer of appropriate type to use with indexing.
    Normally this is a 64-bit signed integer that numpy supports.
    '''
    return np.int64(value)
    
def negative_one(N, modulus = MODULUS):
    '''
    Declare negative unit integer of appropriate type to use with Carmichael transforms.
    Normally this is a 64-bit unsigned integer that numpy supports.
    '''
    return modulus/N-1
    
def negative_integer(N, value, modulus = MODULUS):
    '''
    Declare negative integer of appropriate type to use with Carmichael transforms.
    Normally this is a 64-bit unsigned integer that numpy supports.
    '''
    return modulus/N-value
    
def zeros(shape):
    '''
    Convenience member for defining 64-bit integer arrays with zeros suitable for use with
    Carmichael functions.
    '''
    return np.zeros(shape, dtype=np.uint64)
    
def ones(shape):
    '''
    Convenience member for defining 64-bit integer arrays with zeros suitable for use with
    Carmichael functions.
    '''
    return np.ones(shape, dtype=np.uint64)

def arange(minValue, maxValue, step=1):
    '''
    Convenience member for defining 64-bit integer arrays with zeros suitable for use with
    Carmichael functions.
    '''
    return np.arange(minValue, maxValue, step, dtype=np.uint64)

def randint(minValue, maxValue, shape):
    '''
    Convenience member for defining 64-bit integer arrays with random values upto and excluding
    maximum value provided
    '''
    data = zeros(shape)
    for index, value in enumerate(data):
        data[index] = integer(np.random.randint(minValue, high=maxValue, size=None))
    return data
    
def multiply_1D(fctSignal1, fctSignal2, modulus = MODULUS):
    '''
    Multiply two 1D signals (mod M) and return the result as an array
    '''
    result = zeros(fctSignal1.shape)
    for index, value in enumerate(np.nditer(fctSignal1)):
#        print "index:",index
        val1 = integer(fctSignal1[index])
        val2 = integer(fctSignal2[index])
        result[index] = (val1*val2)%modulus
        
    return result
    
def multiply_2D(array1, array2, modulus = MODULUS):
    '''
    Convenience function to multiply two arrays (mod M)
    '''
    result = zeros(array1.shape)
    for x, row in enumerate(array1):
        for y, col in enumerate(row):
            val1 = integer(array1[x,y])
            val2 = integer(array2[x,y])
            result[x,y] = (val1*val2)%modulus
    
    return result

def bigshr(d, s, n):
    '''
    Shifts s right one bit to d, returns carry bit.
    Adapted from Mikko Tommila et al., APFloat Library 2005.
    '''
    b = integer(0)
    t = integer(0)
    tmp = integer(0)
    
    if not n:
        return integer(0)

    d += n
    s += n

    for t in xrange(0, n):
        d -= integer(1)
        s -= integer(1)

        tmp = (s >> integer(1)) + (0x80000000 if b else integer(0))
        b = s & integer(1)
        d = tmp                              # Works also if d = s

    return b, d

def pow_mod_direct(base, exp, modulus = MODULUS):
    '''
    Compute the power base^exp (mod modulus).
    This computation is as simple and direct as possible.
    '''
    r = integer(1) 
    
    if exp == integer(0):
        return r
        
    for e in arange(0, exp):
        r *= base
        r %= modulus
    
    return r

def pow_mod(base, exp, modulus = MODULUS):
    '''
    Compute the power base^exp (mod modulus).
    Adapted from Mikko Tommila et al., APFloat Library 2005.
    '''
    r = integer(0)    
    b = integer(0) 
#    print 'MODULUS:', modulus, 'base:', base, 'exp:', exp
    
    if exp == integer(0):
        return integer(1)

    b, exp = bigshr(exp, exp, integer(1))
    while not b:
        base = (base * base)%modulus
        b, exp = bigshr(exp, exp, integer(1))

    r = base

    while exp > integer(0):
        base = (base * base)%modulus
        b, exp = bigshr(exp, exp, integer(1))
        if b:
            r = (r * base)%modulus

    return r
    
def ct(data, n, isign = NTT_FORWARD, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Compute the direct Carmichael Transform (i.e. not the Fast version).
    Returns the transformed signal.
    '''
    w = integer(0)    
    
    if isign > 0:
        w = pow_mod_direct(pr, integer(maxLength - maxLength / n), modulus) #n must be power of two
    else:
        w = pow_mod_direct(pr, integer(maxLength / n), modulus)    
    print "Base:", w
    
    result = zeros(n)
    sums = zeros(n)
    for k in arange(0, n):
#        sums[k] = 0
        sums[k] = modulus-n
        for t in arange(0, n):
#            harmonic = pow_mod_direct(w, integer( k*t ), modulus)
            harmonic = pow_mod_direct(w, integer( (2*k+1)*(2*t+1) ), modulus)
            result[k] += ( data[t] * harmonic )%modulus
            result[k] %= modulus
            sums[k] += harmonic
            sums[k] %= modulus
#            print "t:", t, "exp:", integer(k*t), "\tdata[t]:", data[t], "*", harmonic, "=", ( data[t] * harmonic )%modulus, "\tresult[k]:", result[k], "\tsums[k]:", sums[k]
            print "t:", t, "exp:", integer((2*k+1)*(2*t+1)), "\tdata[t]:", data[t], "*", harmonic, "=", ( data[t] * harmonic )%modulus, "\tresult[k]:", result[k], "\tsums[k]:", sums[k]
        print "\n"
    print "Sums:", sums
    print "Result:", result
    
#    for k in arange(1, n):
#        sums[k] = modulus-sums[k]
#        if isign > 0:
#            result[k] = (result[k]-sums[k]+modulus)%modulus
#        else:
#            result[k] = (result[k]+sums[k])%modulus
#        
#    print "Sums Corrected:", sums
    
    return result
    
def rearrange(data, n):
    '''
    Bit-reversal of Data of size n inplace. n should be dyadic.
    Adapted from Mikko Tommila et al., APFloat Library 2005.
    '''
    target = integer(0)
    mask = integer(n)
    
    #For all of input signal
    for position in xrange(0, n):
        #Ignore swapped entries
        if target > position:
            #Swap
            data[position], data[target] = data[target], data[position]

        #Bit mask
        mask = n
        #While bit is set
        mask >>= integer(1)
        while target & mask:
            #Drop bit
            target &= ~mask
            mask >>= integer(1)
        #The current bit is 0 - set it
        target |= mask
    
def fct(data, nn, isign = NTT_FORWARD, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Computes the 1D Fast Carmichael Number Theoretic Transform (FCT) using the Cooley-Tukey algorithm.
    The result is NOT normalised within the function.
    
    Default parameters will work fine for dyadic lengths.
    maxLength is normally modulus-1 or modulus/4 depending on type of modulus.
    pr is normally either 3 or 5 depending on modulus.
    
    Other parameters include:
    2113929217, 3 for lengths upto 2^25 as M=63*2^25+1
    2147473409, 3
    
    The transform is done inplace, destroying the input. 
    '''
    w = wr = wt = integer(0)
    wtemp = integer(0)
    istep = i = m = integer(0)

    if isign > 0:
        w = pow_mod(pr, integer(maxLength - maxLength / nn), modulus) #nn must be power of two
    else:
        w = pow_mod(pr, integer(maxLength / nn), modulus)

    rearrange(data, nn)

    mmax = integer(1)
    while nn > mmax:
        istep = mmax << integer(1)
        wr = wt = pow_mod(w, integer(nn / istep), modulus)

        #Optimize first step when wr = 1
        for i in xrange(0, nn, istep):
            j = i + mmax
            wtemp = data[j]
            data[j] = (data[i] + modulus - wtemp) if data[i] < wtemp else (data[i] - wtemp)
#            data[j] = (data[i] - wtemp)%modulus #causes underflow sometimes
            data[i] = (data[i] + wtemp)%modulus

        for m in xrange(1, mmax):
            for i in xrange(m, nn, istep):
                j = i + mmax
                wtemp = (wr * data[j])%modulus #double width for integer multiplication
                data[j] = (data[i] + modulus - wtemp) if data[i] < wtemp else (data[i] - wtemp)
#                data[j] = (data[i] - wtemp)%modulus #causes underflow sometimes
                data[i] = (data[i] + wtemp)%modulus 
            wr = (wr * wt)%modulus #double width for integer multiplication
        mmax = istep
        
    return data
    
def fct_2D(data, nn, isign = NTT_FORWARD, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Computes the 2D Carmichael transform for a 2D square array of size nn
    '''
#    d = euclidean((nttw_big_integer)nn,MODULUS,&inv,&y); #Multi Inv of p-1
#    inv = (inv + MODULUS)%MODULUS; #Ensure x is positive

    #Transform Rows
    for row in data: #done in place
        fct(row, nn, isign, pr, modulus, maxLength)

    #Transform Columns
    for column in data.T: #done in place, transpose is cheap
#        for (k = 0; k < nn; k ++)
#            ptrResult[k] = (data[k*nn+j] * inv)%MODULUS; #Stops modulo overrun, div by N early

        fct(column, nn, isign, pr, modulus, maxLength)

#        for (k = 0; k < nn; k ++) #Inverse so Copy and Norm
#            result[k*nn+j] = ptrResult[k];
    return data
    
def ifct(data, nn, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Convenience function for inverse 1D Fast Carmichael Transforms. See fct documentation for more details.
    '''
    return fct(data, nn, NTT_INVERSE, pr, modulus, maxLength)

def ifct_2D(data, nn, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Convenience function for inverse 2D Fast Carmichael Transforms. See fct_2D documentation for more details.
    '''
    return fct_2D(data, nn, NTT_INVERSE, pr, modulus, maxLength)
    
def harmonics(n, isign = NTT_FORWARD, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Computes the harmonics of the 1D Carmichael Number Transform (FCT)
    The returned array then contains the basis functions for each k for the transform.
    '''
    w = integer(0)    
    
    if isign > 0:
        w = pow_mod_direct(pr, integer(maxLength - maxLength / n), modulus) #nn must be power of two
    else:
        w = pow_mod_direct(pr, integer(maxLength / n), modulus)    
    print "Base:", w
    
#    x = np.random.randint(0, n, n)
#    x = arange(0, n)
#    print "x:", x
    
    result = zeros(n)
    harmonics = zeros( (n, n) )
    for k in arange(0, n):
        for t in arange(0, n):
#            value = integer( x[t]*pow_mod(w, integer(k*t), modulus) )
            value = pow_mod_direct(w, integer(k*t), modulus)
#            value = pow_mod_direct(w, integer((k+1)*(t+1)), modulus)
#            value = pow_mod(w, integer(k*t), modulus)
            value %= modulus
#            if value in harmonics[k]:
#                print "Error: Harmonic not unique"
            harmonics[k, t] = value
            result[k] += value
            result[k] %= modulus
#            print "k, harmonic k: ", k, harmonics[k, t]
#            print "k, t, kt, powmod =", k, t, integer(k*t), pow_mod_direct(w, integer(k*t), modulus)
#        print harmonics[k]
#        print "sum, freq:", harmonics[k].sum(), harmonics[k].sum()%modulus
        
    return result, harmonics
    
def norm_1D(data, N, modulus = MODULUS):
    '''
    Normalise the signal given a full forward and inverse transform (mod M)
    '''
    normData = zeros(N)
    Ninv = integer(nt.minverse(N, modulus)%modulus)
    
    if modulus%2 == 1: #odd number then, modulus likely prime and therefore a field
        for i, value in enumerate(np.nditer(data)):
            normData[i] = (value*Ninv)%modulus
    else: #only ring
        for i, value in enumerate(np.nditer(data)):
            normData[i] = value/N
        
    return normData
    
def norm_2D(data, N, modulus = MODULUS):
    '''
    Normalise the signal given a full forward and inverse transform (mod M).
    Assumes data is a square array of size N
    '''
    normData = norm_1D(data.flatten(), N*N, modulus)
        
    return normData.reshape((N,N))

def toPixelRange(data, N, modulus = MODULUS):
    '''
    Renormalise the gray scales so that it is easily displayed.
    '''
#    maxValue = int(data.max())
    intData = data.astype(np.int64)
    intData[intData>(modulus/2)] -= modulus
    
    return intData