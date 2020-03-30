'''
Module for computing the partition of integers

'''
import numpy as np
import mojette

def partitionProjection(proj, angle, P, Q):
    '''
    Given a projection of a PxQ image, work out all the partitions possible for each bin in that projection
    Samples (i.e. k) for each bin is computed internally.
    '''
    onesImage = np.ones((P,Q)) #all ones
    
    p = int(angle.imag)
    q = int(angle.real)
    samples = mojette.project(onesImage, q, p, np.int32)
    
    partitions = []
    for bin, k in zip(proj, samples):
        p = partitionWithLength(bin, k)
        partition = []
        for part in p:
            partition.append(part)
        print "Partition for bin", bin, "with", k, ":", partition
        partitions.append(partition)
    
    return partitions

from itertools import islice #fast list iteration

def partitionReconstruction(projs, angles, P, Q):
    '''
    Given a set of projections of a PxQ image, reconstruct that image using the intersection
    of the partition sets of the projection bins.
    '''
    projPartitions = []
    for proj, angle in zip(projs, angles):
        partitions = partitionProjection(proj, angle, P, Q)
        projPartitions.append(partitions)
    
    #for each pixel, compute the inter section of partitions to recover pixel    
    recon = np.zeros((Q,P))
    for x in range(0, Q):
        for y in range(0, P):
            pixelParts = []
            for part, angle in zip(projPartitions, angles):
                p = int(angle.imag)
                q = int(angle.real)
        
                offsetMojette = 0
                if q*p >= 0: #If positive slope
                    offsetMojette = p*(Q-1)
                    
                if q*p >= 0:
                    translateMojette = q*y - p*x + offsetMojette #GetY = q, GetX = p
                else:
                    translateMojette = p*x - q*y; #GetY = q, GetX = p
                pixelParts.append(part[translateMojette])
            '''
            Intersection of sets fast
            >>> from itertools import islice
            >>> set.intersection(set(lis[0]), *islice(lis, 1, None))
            set([1, 3])
            '''
            print "Pixel Partitions:", pixelParts
            #fast iteration of partition list for pixel and compute intersection of partitions as sets
            result = set.intersection(set(pixelParts[0]), *islice(pixelParts, 1, None))
            print "Result", x, y, ":", result
            recon[x,y] = int(result)
    
    return recon