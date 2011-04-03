'''
Created on Aug 6, 2010

@author: joel
'''
import numpy
import numpy as np
import numpy.linalg as nl
import math
from pypwdg.parallel.decorate import distribute, immutable
import pypwdg.utils.mappings as pum
        
@immutable
class MeshQuadratures(object):
    """ Given a quadrature rule on a reference simplex, provide quadratures on all the faces in a mesh """
    def __init__(self, mesh, quadrule):
        self.__mesh = mesh
        self.__qp, self.__qw = quadrule
         
    
    def quadpoints(self, faceid):
        """ return the quadrature points on face faceid"""
        dirs = self.__mesh.directions[faceid]
#        return numpy.tile(dirs[0], (len(self.__qp),1)) + numpy.dot(self.__qp, dirs[1:-1])
        return dirs[0] + numpy.dot(self.__qp, dirs[1:-1])
    
    def quadweights(self, faceid):
        """ return the quadrature weights on face faceid"""
        return self.__qw * self.__mesh.dets[faceid]

class MeshElementMaps(object):
    """ For any mesh element, return the affine map that maps to it from the reference simplex"""
    def __init__(self, mesh):
        self.mesh = mesh
    
    def getMap(self, e):
        # happily, mesh.directions contains the origin and offsets to all the vertices on the *element* 
        # for each face.  So just pick the first face associated with this element
        dirs = self.__mesh.directions[self.__mesh.etof[e][0]]
        return pum.Affine(dirs[0], dirs[1:])
    

class MeshElementQuadratures(object):
    def __init__(self, mesh, quadrule):
        self.__mesh = mesh
        self.__qp, self.__qw = quadrule
        
    def quadpoints(self, eltid):
        """ return the quadrature points on element eltid"""
        # happily, mesh.directions contains the origin and offsets to all the vertices on the *element* 
        # for each face.  So just pick the first face associated with this element

        dirs = self.__mesh.directions[self.__mesh.etof[eltid][0]]
        return dirs[0] + numpy.dot(self.__qp, dirs[1:])

    def quadweights(self, eltid):
        """ return the quadrature weights on face faceid"""
        # The area of a simplex is 1/n! times the area of a parallepiped;
        area = abs(nl.det(self.__mesh.directions[self.__mesh.etof[eltid][0]][1:]))# / math.factorial(self.__mesh.dim))
        return self.__qw * area

def elementcentres(mesh):
    # add the first direction to the average of the others (the offsets)
    x = np.concatenate([[1], np.ones(mesh.dim)*1.0/(mesh.dim+1)])
    for fs in mesh.etof: print mesh.directions[fs[0]] 
    
    # pick the first face associated with each element
    return [np.dot(x,mesh.directions[fs[0]]) for fs in mesh.etof]  

