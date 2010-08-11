'''
Created on Aug 11, 2010

@author: joel
'''
import numpy
import scipy.sparse

def pointsToElement(points, mesh, SM):
    """ detect which element each point is in """
    
    # take the dot product with each point and the outward normal on each face
    pointsn = numpy.dot(points, mesh.normals.transpose())

    # now do the same thing for an (arbitrary) point on each face
    originsn = numpy.sum(mesh.directions[:,0,:] * mesh.normals, axis = 1).reshape((1,-1))
    
    # this gives the distance of each point from each face    
    offsets = pointsn - originsn
    
    # normals point outwards, so detect when the distance is non-positive
    behindface = offsets <=0
    
    # for each element, sum over the neighbouring faces.  detect which point is behind dim+1 faces
    withinelt = scipy.sparse.csr_matrix((behindface * SM.eltstofaces) == (mesh.dim + 1))
    
    # for each point, pick an element that it's a member of (this is arbitrary for points that lie on faces)
    return withinelt.indices[withinelt.indptr[:-1]]
            

    
    