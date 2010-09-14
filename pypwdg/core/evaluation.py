'''
Created on Aug 11, 2010

@author: joel
'''

from pypwdg.core.vandermonde import ElementVandermondes
from pypwdg.mesh.structure import StructureMatrices
from pypwdg.utils.geometry import pointsToElementBatch
from pypwdg.utils.timing import print_timing
from pypwdg.parallel.decorate import distribute, partitionlist


import numpy

@distribute(lambda n: lambda mesh,eltobasis,points : [((mesh,eltobasis,pp),{}) for pp in partitionlist(n,points)])
class Evaluator(object):
    def __init__(self, mesh, elttobasis, points):
        self.mesh = mesh
        self.points = points
        SM = StructureMatrices(mesh)
        ptoe = pointsToElementBatch(points, mesh, SM, 5000)
        # could use sparse matrix classes to speed this up, but it's a bit clearer like this
        # pointsToElement returns -1 for elements which have no point
        self.etop = [[] for e in range(mesh.nelements+1)] 
        for p,e in enumerate(ptoe):
            self.etop[e+1].append(p)
        
        self.v = ElementVandermondes(mesh, elttobasis, lambda e: points[self.etop[e+1]])
    
    def evaluate(self, x):
        vals = numpy.zeros(len(self.points), dtype=numpy.complex128)
        n = 0
        for e,p in enumerate(self.etop[1:]):
            nb = self.v.numbases[e]
#            print self.v.getVandermonde(e)
#            print x[n:n+nb]
            
            vals[p] += numpy.dot(self.v.getVandermonde(e), x[n:n+nb])
            n+=nb
#            print "Vals = %s"%vals
        
        return vals
        