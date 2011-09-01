'''
Created on Aug 31, 2011

@author: joel
'''
import scipy.optimize as so
import scipy.interpolate as si
import numpy as np

def nearestpoint(point, surface):
    ''' returns the parameters giving the nearest point on surface to point'''
    s = so.fmin_powell(lambda s: np.sum((surface(s) - point)**2), np.zeros(len(point)-1))[0]

class FaceMapping:
    ''' Given a surface and a flat face, represents the mapping from the face to the surface
    
        surface should be a callable representing a chart mapping.  surface need not be a unit
        speed parameterisation (indeed, that is, in general, impossible when surface is a manifold of 
        dimension greater than 1).  If it is not, a piecewise linear mapping is inferred from the 
        chart domain to the face. 
        
        To evaluate the map, points that do not lie on the face are first projected onto the face 
        along the line that passes through othervertex 
    '''
    
    def __init__(self, facevertices, othervertex, surface, interpolationpoints = 10):
        dim = len(facevertices)
        facesurfparams = map(nearestpoint, facevertices)
        faceparaminterpolationpoints = 
         
        
        