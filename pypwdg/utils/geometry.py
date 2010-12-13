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
        npoints is an array containing the number of points in each direction
    """
    
    def __init__(self, bounds, npoints):
        self.lower = np.min(bounds, 0) # self.lower is the most negative vertex
        self.upper = np.max(bounds, 0) # self.upper is the most positive vertex
        self.npoints = npoints
        self.strides = [np.prod(npoints[i+1:]) for i in range(len(npoints))]
        self.dim = bounds.shape[1]
        self.length = np.prod(npoints)
    
    def getPoints(self, vertices):
        """ Returns a tuple (idxs, points) where points contains all the points that are inside
            the convex hull of vertices and idxs are their corresponding global indices
            """ 
        intervals = self.npoints - 1    
        elower = np.max((self.lower, np.amin(vertices,axis = 0)),0) # find the negative vertex of ebounds
        eupper = np.min((self.upper, np.amax(vertices,0)),0) # find the positive vertex of ebounds
        
        # find the lower and upper bounds for the indices that we need
        lower = np.floor(intervals * (elower - self.lower) / (self.upper - self.lower)).astype(int)
        upper = np.ceil(intervals * (eupper - self.lower) / (self.upper - self.lower)).astype(int)+1
        
        # We're going to take advantage of numpy array broadcasting and assemble a hypercube.
        # of indices and points.  The first step is to work out how to reshape the indices in each
        # axis.        
        shapes = np.ones((self.dim, self.dim)) - 2*np.eye(self.dim)
        axisidxs = [np.arange(l,u).reshape(shape) for l,u,shape in zip(lower, upper, shapes)]
        idxs = sum([axisidx * stride for axisidx, stride in zip(axisidxs, self.strides)])
        points = np.zeros(idxs.shape + (self.dim,))
        # A for loop.  Kill me now.
        for i, l, u, n in zip(range(self.dim), self.lower, self.upper, self.npoints):
            points[...,i]+=(axisidxs[i]*(u-l)/n + l)
            
        return idxs.ravel(), points.reshape((-1,self.dim))
        
def elementToStructuredPoints(structuredpoints, mesh, eid):
    vertices = mesh.elements[eid]
    crudeidxs, crudepoints = structuredpoints.getPoints(mesh.nodes[vertices])
    fs = mesh.etof[eid]
    normals = np.array([mesh.normals[f] for f in fs])
    directions = np.array([mesh.directions[f] for f in fs])
    # take the dot product with each point and the outward normal on each face
    pointsn = numpy.dot(crudepoints, normals.transpose())

    # now do the same thing for an (arbitrary) point on each face
    originsn = numpy.sum(directions[:,0,:] * normals, axis = 1).reshape((1,-1))
    
    # this gives the distance of each point from each face    
    offsets = pointsn - originsn
    # normals point outwards, so detect when the distance is non-positive
    behindface = offsets <=0
#    if (offsets==0).any(): print "Warning, points on the boundary"
    inelement = (np.sum(behindface, axis=1)==len(fs))
    
    return crudeidxs[inelement], crudepoints[inelement]
    
            
        
            
        
        
        
    