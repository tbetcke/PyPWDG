'''
Created on Aug 24, 2011

@author: joel
'''
import pypwdg.core.bases as pcb
import pypwdg.core.bases.variable as pcbv
import pypwdg.core.bases.reference as pcbr
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import test.utils.mesh as tum
import numpy as np

class harmonic1():
    def values(self, x):
        return (x[:,0]**2 - x[:,1]**2).reshape(-1,1)
    def gradient(self, x):
        return x * [2,-2] 

class HarmonicDerived(pcb.Basis):
    def __init__(self, k, S):
        self.k = k
        self.S = S
        
    def values(self, x):
        return np.exp(1j * self.k * self.S.values(x))

    def derivs(self, x, n=None):
        if n is None:
            return (self.S.gradient(x) * self.values(x))[:,np.newaxis,:]
        else:
            return np.dot(self.S.gradient(x), n)[:,np.newaxis] * self.values(x)
    
    def laplacian(self, x):
        return -self.k**2 * self.values(x)

import pypwdg.parallel.main

from numpy import array,sqrt

k = 20
S = harmonic1()

direction=array([[1.0,1.0]])/sqrt(2)
g = HarmonicDerived(k, S)

bdytag = "BDY"

#bnddata={bdytag:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
bnddata={bdytag:pcbd.dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([100,100])

n = 20
mesh = tum.regularsquaremesh(n, bdytag)

basisrule = pcbv.PlaneWaveVariableN(pcb.circleDirections(12))
#basisrule = pcbr.ReferenceBasisRule(pcbr.Dubiner(3))


entityton ={1:lambda x: np.sqrt(np.sum(S.gradient(x)**2, axis=1))}
problem=psp.VariableNProblem(entityton, mesh,k, bnddata)
#problem = psp.Problem(mesh, k, bnddata)


computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, 15)
solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
pos.comparetrue(bounds, npoints, g, solution)
pos.standardoutput(computation, solution, 20, bounds, npoints, 'pwconvergence')
