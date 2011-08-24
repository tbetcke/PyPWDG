'''
Created on Aug 24, 2011

@author: joel
'''
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.raytrace.control as prc
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
        d = self.S.gradient(x)[:,np.newaxis, :] if n is None else np.dot(self.S.gradient(x), n)
        return 1j * self.k * d * self.values(x)
    
    def laplacian(self, x):
        return -self.k**2 * self.values(x)

import pypwdg.parallel.main

from numpy import array,sqrt

k = 10
S = harmonic1

direction=array([[1.0,1.0]])/sqrt(2)
g = HarmonicDerived(k, harmonic1())

bdytag = "BDY"

bnddata={bdytag:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([100,100])

n = 3
mesh = tum.regularsquaremesh(n, bdytag)

problem = psp.Problem(mesh, k, bnddata)
computation = psc.Computation(problem, pcb.planeWaveBases(2,k,11), pcp.HelmholtzSystem, 15)
solution = computation.solution(psc.DirectSolver().solve)
pos.standardoutput(computation, solution, 20, bounds, npoints, 'pwconvergence')
