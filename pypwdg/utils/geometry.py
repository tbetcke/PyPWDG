'''
Created on Aug 11, 2010

@author: joel
'''

import numpy
import scipy.sparse
import numpy.ma as ma
import numpy as np

def pointsToElement(points, mesh):
    """ detect which element each point is in """
    
    normals = mesh.normals[mesh.fs]
    directions = mesh.directions[mesh.fs]
    assert(ma.count_masked(normals)==0)
#    normals = numpy.array(normals)
    
    # take the dot product with each point and the outward normal on each face
    pointsn = numpy.dot(points, normals.transpose())

    # now do the same thing for an (arbitrary) point on each face
    originsn = numpy.sum(directions[:,0,:] * normals, axis = 1).reshape((1,-1))
    
    # this gives the distance of each point from each face    
    offsets = pointsn - originsn
    
    # normals point outwards, so detect when the distance is non-positive
    behindface = offsets <=0
    
    # for each element, sum over the neighbouring faces.  detect which point is behind dim+1 faces
    withinelt = scipy.sparse.csr_matrix((behindface.data * mesh.elttofaces.transpose()[mesh.fs]) == (mesh.dim + 1))
    
    if len(withinelt.indices):
        # for each point, pick an element that it's a member of (this is arbitrary for points that lie on faces)
        ptoe = withinelt.indices[withinelt.indptr[:-1] % len(withinelt.indices)]
        
        # If the point is not in any element, assign it to -1
        ptoe[withinelt.indptr[:-1]==withinelt.indptr[1:]] = -1
    else:
        ptoe = numpy.zeros(len(points), dtype=int)-1
    
    return ptoe

def pointsToElementBatch(points, mesh, batchsize = 5000):    
    return numpy.concatenate([pointsToElement(points[i*batchsize:min(len(points),(i+1)*batchsize)], mesh) for i in range((len(points)-1)/batchsize+1)])        

class StructuredPoints(object):
    """ Structured points in a hypercube.  
    
        bounds[0] and bounds[1] should be opposite vertices of the hypercube.
        npoints is the number of points in each direction
    """
    
    def __init__(self, bounds, npoints):
        self.lower = np.min(bounds, 0) # self.lower is the most negative vertex
        self.upper = np.max(bounds, 0) # self.upper is the most positive vertex
        self.npoints = npoints
    
    def getPoints(self, ebounds):
        """ Returns a tuple (idxs, points) where points are all the points lying within 
            the hypercube specified by ebounds and idxs are their corresponding global indices""" 
        intervals = self.npoints - 1    
        elower = np.max(self.lower, np.min(ebounds,0)) # find the negative vertex of ebounds
        eupper = np.min(self.upper, np.max(ebounds,0)) # find the positive vertex of ebounds
        lower = np.floor(intervals * (elower - self.lower) / (self.upper - self.lower)).astype(int)
        upper = np.ceil(intervals * (eupper - self.lower) / (self.upper - self.lower)).astype(int)+1
        
        
        
        
    