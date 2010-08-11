'''
Created on Aug 6, 2010

@author: joel
'''
import numpy
        
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
