'''
Created on Aug 11, 2010

@author: joel
'''

from pypwdg.core.vandermonde import LocalInnerProducts
from pypwdg.utils.geometry import pointsToElementBatch, elementToStructuredPoints
from pypwdg.utils.timing import print_timing
from pypwdg.parallel.decorate import parallelmethod, distribute, tuplesum
from pypwdg.mesh.meshutils import MeshQuadratures

import numpy

import pypwdg.mesh.structure as pms
import pypwdg.utils.sparse as pus
import scipy.sparse as ss
import numpy as np
import pypwdg.core.vandermonde as pcv


@distribute()
class Evaluator(object):
    @print_timing
    def __init__(self, mesh, elttobasis, points):
        self.mesh = mesh
        self.points = points
        ptoe = pointsToElementBatch(points, mesh, 5000)
        # could use sparse matrix classes to speed this up, but it's a bit clearer like this
        # pointsToElement returns -1 for elements which have no point
        self.etop = [[] for e in range(mesh.nelements+1)] 
        for p,e in enumerate(ptoe):
            self.etop[e+1].append(p)
        self.elttobasis = elttobasis        
#        self.v = ElementVandermondes(mesh, elttobasis, lambda e: points[self.etop[e+1]])
    
    @parallelmethod()
    @print_timing
    def evaluate(self, x):
        vals = numpy.zeros(len(self.points), dtype=numpy.complex128)
        for e,p in enumerate(self.etop[1:]):
            v = self.elttobasis.getValues(e, self.points[p])            
            (vidx0,vidx1) = self.elttobasis.getIndices()[e:e+2]
            vals[p] += numpy.dot(v, x[vidx0: vidx1])
        return vals


@distribute()
class StructuredPointsEvaluator(object):
    """ Allows the evaluation of a function at some StructuredPoints.  The function
        is determined by a basis and a corresponding vector of coefficients
        
        See pypwdg.utils.geometry for the StructuredPoints class        
    """        
    def __init__(self, mesh, elttobasis, filter, x):
        self.mesh = mesh
        self.elttobasis = elttobasis
        self.filter = filter
        self.x = x
    
    @parallelmethod(reduceop = tuplesum)
    def evaluate(self, structuredpoints):
        outputshape = structuredpoints.length if len(self.x.shape) == 1 else (structuredpoints.length, self.x.shape[1])
        vals = numpy.zeros(outputshape, dtype=numpy.complex128)
        pointcount = numpy.zeros(structuredpoints.length, dtype=int)
        for e in self.mesh.partition:
            pointidxs, points = elementToStructuredPoints(structuredpoints, self.mesh, e)
            if len(pointidxs):
                v = self.elttobasis.getValues(e, points)
                (vidx0,vidx1) = self.elttobasis.getIndices()[e:e+2]       
                vals[pointidxs] += numpy.dot(v, self.x[vidx0: vidx1])
                pointcount[pointidxs]+=1
        return self.filter(vals), pointcount
    




        
        