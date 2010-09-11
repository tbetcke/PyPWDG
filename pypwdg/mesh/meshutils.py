'''
Created on Aug 6, 2010

@author: joel
'''
import numpy
import numpy.linalg as nl
import math
        
class MeshQuadratures(object):
    """ Given a quadrature rule on a reference simplex, provide quadratures on all the faces in a mesh """
    def __init__(self, mesh, quadrule):
        self.__mesh = mesh
        self.__qp, self.__qw = quadrule
        
    def quadpoints(self, faceid):
        """ return the quadrature points on face faceid"""
        dirs = self.__mesh.directions[faceid]
        return numpy.tile(dirs[0], (len(self.__qp),1)) + numpy.dot(self.__qp, dirs[1:-1])

    def quadweights(self, faceid):
        """ return the quadrature weights on face faceid"""
        return self.__qw * self.__mesh.dets[faceid]

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
        area = abs(nl.det(self.__mesh.directions[self.__mesh.etof[eltid][0]][1:]) / math.factorial(self.__mesh.dim))
        return self.__qw * area 
    