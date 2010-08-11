'''
Created on Aug 11, 2010

@author: joel
'''

from PyPWDG.core.vandermonde import ElementVandermondes
from PyPWDG.mesh.structure import StructureMatrices
from PyPWDG.utils.geometry import pointsToElement
from PyPWDG.utils.timing import print_timing

import numpy

class Evaluator(object):
    @print_timing
    def __init__(self, mesh, elttobasis, points):
        self.mesh = mesh
        self.points = points
        SM = StructureMatrices(mesh)
        ptoe = pointsToElement(points, mesh, SM)
        # could use sparse matrix classes to speed this up, but it's a bit clearer like this
        self.etop = [[] for e in range(mesh.nelements)] 
        for p,e in enumerate(ptoe):
            self.etop[e].append(p)
        
        self.v = ElementVandermondes(mesh, elttobasis, lambda e: points[self.etop[e]])
    
    @print_timing    
    def evaluate(self, x):
        vals = numpy.zeros(len(self.points), dtype=numpy.complex128)
        n = 0
        for e,p in enumerate(self.etop):
            nb = self.v.numbases[e]
#            print self.v.getVandermonde(e)
#            print x[n:n+nb]
            
            vals[p] += numpy.dot(self.v.getVandermonde(e), x[n:n+nb])
            n+=nb
#            print "Vals = %s"%vals
        
        return vals
        