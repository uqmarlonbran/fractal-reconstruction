# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:00:40 2019

Generate Fractals for different image shapes and Farey order

@author: marlo
"""
import os
import sys
import matplotlib.pylab as plt
plt.switch_backend('Agg')
sys.path.insert(0, './fareyFunctions')
from fareyFractal import farey_fractal

def prime_list(givenNumber):  
    
    # Initialize a list
    primes = []
    for possiblePrime in range(2, givenNumber + 1):        # Assume number is prime until shown it is not. 
        isPrime = True
        for num in range(2, int(possiblePrime ** 0.5) + 1):
            if possiblePrime % num == 0:
                isPrime = False
                break        
        if isPrime:
            primes.append(possiblePrime)
    
    return(primes)
def truncate(n):
    return int(n * 1000) / 1000

# Go through prime numbers and generate some fractals
primeArrayTemp = prime_list(601)
primeArray = []
for x in primeArrayTemp:
    if x > 50:
        primeArray.append(x)

for N in range(4, 13):
    print(N)
    # Create Directory
    dirName = "Experiments/" + str(N)
    
    try:
        # Create target directory
        os.mkdir(dirName)
        print("Directory " , dirName, " Created.")
    except FileExistsError:
        print("Directory " , dirName, " already exists.")
    
    for p in primeArray:
        lines, angles, m, fractal, R, oversampleFilter = farey_fractal(p, N, centered=True, twoQuads=True)
        string = "(Prime, N) = (" + str(p) + "," + str(N) + ") " + " R = " + str(truncate(R))
        fig = plt.figure()
        fig.suptitle(string, fontsize=20)
        plt.imshow(fractal)
        saveDir = dirName + "/" + str(p) + ".jpg"
        fig.savefig(saveDir, dpi=300)
        plt.close(fig)


