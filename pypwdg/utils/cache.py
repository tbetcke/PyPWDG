'''
Created on Apr 5, 2011

@author: joel
'''

import numpy as np

class PointDict(object):
    """ A dictionary with points (1-dimensional arrays) as keys and arrays as values.
    The intent is to use it as a cache so lookup is fast, insertion is pretty slow.
    
    In particular, it works with, and is optimised for, look-ups based on multiple points.
    
    The fast lookup is based on mapping all points into a known hypercube, so it's important to have 
    an idea of how the points are distributed (although arctan could be used as the mapping to support
    caching over all of Euclidean space)  
    
    dim: the dimension of the arrays
    shape: the shape of the value arrays
    valuetype: the type of the value arrays 
    map: an optional function that maps arrays into a [0,1]^dim hypercube.
    N: the number of buckets, i.e. the number of buckets in each dimension will be roughly N^(1/dim)
    eps: a threshold for the components of points to be equal (after mapping)
    """    
    def __init__(self, dim, shape = (1,), valuetype=float, map = None, buckets = 1000000, eps = 1E-10, ):
        N1 = int(buckets**(1.0/dim))
        self.buckets = np.ones((N1, N1), dtype=int)*-1 
        maxsize = buckets/2        
        self.points = np.empty(maxsize, dim, dtype=float)
        self.values = np.empty((maxsize,)+shape, dtype = valuetype)        
        self.size = 0
        self.map = map
        
    def lookup(self, p):
        pmap = self.map(p) if map else p
        pbucketidx = np.floor(pmap * self.buckets.shape)
        pbucket = self.buckets[[pbucketidx[:,i] for i in range(p.shape[1])]] 
        pbucket1 = pbucket[pbucket >= 0]
        pointmatch = np.prod(np.abs(self.points[pbucket1] - pmap), axis=1, dtype=bool)
        if  
        
        
        