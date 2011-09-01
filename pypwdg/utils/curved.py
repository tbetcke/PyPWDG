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
    return s

def uniformreferencepoints(dim, n):
    a = np.linspace(0,1,n)
    p = np.array([np.tile(a.reshape(1 - 2*s), n*(1-s)+s) for s in np.eye(dim)]).reshape(dim,-1).transpose() # This produces a grid of points on the unit hypercube of dimension dim (I promise!)
    return p[p.sum(1) <=1] # We only want the points on the unit simplex
    
    

class FaceMapping:
    ''' Given a simplex, an identification of a subsimplex (vertex, edge or face) and a surface,
        maps the whole simplex so that the subsimplex now lies on the surface
    
        surface should be a callable representing a chart mapping, s: U \subset R^{n-1} \rightarrow S \subset R^n.  
        
        The face, F is given by its vertices.  FaceMapping attempts to determine a map p:F \rightarrow U such that
        for any x \in F
        
        f := s \circ p : F \rightarrow S minimises \abs{f(x) - x}    
         
        It's expensive to solve this optimisation problem for every x, so FaceMapping in fact determines p for a
        given array of points and then uses a piecewise linear interpolation.
        
        To evaluate the map, points that do not lie on the face are first projected onto the face 
        along the line that passes through othervertex.
        
        TODO: 
        - Allow for other "nearest point" functions.
        - Allow for other interpolation options for p         
    '''
    
    def __init__(self, vertices, subsimplex, surface, interpolationpoints = 10):
        
        dim = len(vertices)-1
        ssdim = len(subsimplex)-1
        self.barycentric = np.linalg.inv(np.hstack((vertices, np.ones(len(vertices),1))))
        
        # The documentation lies a bit.  In fact, we're going to determine pp : FF \rightarrow U, where
        # FF is the reference simplex and so f = s \circ pp \circ \phi^{-1}, where \phi: FF \rightarrow F
        # is a chart mapping for the flat face, F 
        if ssdim >0:
            refpoints = uniformreferencepoints(ssdim-1, interpolationpoints) # Get a grid of points on the reference simplex 
            barycentricrefpoints = np.hstack(refpoints, 1-np.sum(refpoints, axis=1).reshape(-1,1))
            sspoints = np.dot(barycentricrefpoints, vertices[:,subsimplex])
            
            sssurfparams = np.array(map(nearestpoint, sspoints)) # For each point, x \in F, find the point in y \in U that such that s(y) minimises |s(y) - x|
            self.p = [si.LinearNDInterpolator(refpoints, params) for params in sssurfparams.transpose()]
        else:
            y = nearestpoint(vertices[subsimplex])
            self.p = lambda x : y
        
        self.vertices = vertices
        self.subsimplex = subsimplex
        
    def apply(self, x):
        barycoords = np.dot(x, self.barycentric)
        
        
        
         
        
        