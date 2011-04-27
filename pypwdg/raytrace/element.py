'''
Created on Mar 8, 2011

@author: joel
'''
import numpy as np

def intersect(linepoint, linedir, planepoint, planedirs):
    ''' Determine where the line given by linepoint + \mu * linedir intersects the plane given by 
    planepoint + \lambda planedirs, where \mu \in R and \lambda \in R^n, where n is the dimension of the plane
    
    Returns (\lambda, \mu) '''
    
    M = np.vstack((planedirs, -linedir))
    if np.linalg.det(M) == 0: return (None, np.inf) 
    P = linepoint - planepoint
    x = np.linalg.solve(M.transpose(), P.transpose()).ravel()
    return (x[:-1], x[-1])
     

def reflect(linedir, planedirs):
    ''' Reflect linedir in the plane spanned by planedirs'''
    Q, R = np.linalg.qr(np.vstack((planedirs, linedir)).transpose())
    r = R[:,-1] # the final column of R gives the coefficients of linedir in the columns of Q
    r[-1]*=-1 # the final column of Q is orthogonal to the plane.  Lets reverse the direction
    return np.dot(Q,r) # et voila.
    
class HomogenousTrace(object):
    ''' Knows how to trace through one element assuming homogenous material data''' 
    
    def __init__(self, mesh, nonreflecting):
        self.mesh = mesh
        self.reflectingfaces = mesh.faceentities != [None]
        print mesh.faceentities != None
        for e in nonreflecting:
            self.reflectingfaces[mesh.faceentities == e] = False        
        self.neighbourface = mesh._connectivity * np.arange(1, mesh.nfaces+1, dtype=int) - 1
                
    def trace(self, face, point, direction):
        ''' Given an input face, a point on the face and a direction trace a path through the relevant element
        
        Returns: (e, f, p, d) where:
            e is the element that was traced through, 
            f is the next input face
            p is the next point
            d is the next direction'''
        
        e = self.mesh.ftoe[face]
        fs = self.mesh.etof[e]
        
        for f in fs:
            if f==face: continue
            dirs = self.mesh.directions[f]
            planedirs = dirs[1:self.mesh.dim]
            l, m = intersect(point, direction, dirs[0], planedirs)
            if l is not None and (l > 0).all() and sum(l) <= 1 and m > 0: # assumes that the face is a simplex
                p = point + m * direction
                if self.reflectingfaces[f]:
                    direction = reflect(direction, planedirs)
                else:
                    f = self.neighbourface[f]
                return (e,f,p,direction)
        
#        print "Failed to find face to trace ",face,point,direction,e
        return None
        
        
         
