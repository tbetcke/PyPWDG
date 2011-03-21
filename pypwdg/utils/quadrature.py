'''
Created on Aug 4, 2010

@author: joel
'''

import scipy.special.orthogonal
import numpy
    
def trianglequadrature(n):
    """ Degree n quadrature points and weights on a triangle (0,0)-(1,0)-(0,1)"""

    x00,w00 = scipy.special.orthogonal.p_roots(n)
    x01,w01 = scipy.special.orthogonal.j_roots(n,1,0)
    x00s = (x00+1)/2
    x01s = (x01+1)/2
    w = numpy.outer(w01, w00).reshape(-1,1) / 8 # a factor of 2 for the legendres and 4 for the jacobi10
    x = numpy.outer(x01s, numpy.ones(x00s.shape)).reshape(-1,1)
    y = numpy.outer(1-x01s, x00s).reshape(-1,1)
    return numpy.hstack((x, y)), w

def tetquadrature(n):
    txy, tw = trianglequadrature(n)
    x02,w02 = scipy.special.orthogonal.j_roots(n,2,0)
    x02s = (x02+1)/2
    xy = ((1-x02s).reshape(-1,1,1) * txy.reshape(1,2,-1)).reshape(-1,2) 
    z =  numpy.repeat(x02s,len(txy)).reshape(-1,1)
    w = (tw.reshape(1,-1) * w02.reshape(-1,1)).reshape(-1,1)/8
    return numpy.hstack((xy, z)), w
    
def legendrequadrature(n):
    """ Legendre quadrature points on [0,1] """
    x00,w00 = scipy.special.orthogonal.p_roots(n)
    return (x00.reshape(-1,1)+1)/2, w00/2

def squarequadrature(n):
    x00,w00 = legendrequadrature(n)
    g = numpy.mgrid[0:n,0:n].reshape(2,-1)
    w = w00[g[0]] * w00[g[1]]
    x = numpy.hstack((x00[g[0]], x00[g[1]]))
    return x, w
