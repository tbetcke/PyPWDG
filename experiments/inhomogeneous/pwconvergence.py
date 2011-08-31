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
import pypwdg.output.mploutput as pom
import pypwdg.test.utils.mesh as tum
import numpy as np
import math
import matplotlib.pyplot as mp
import gc

class harmonic1():
    ''' Harmonic function s * ((x+t)^2 - (y+t)^2), with s and t chosen such that the gradient has length 1 at (0,0) and self.scale at (1,1)''' 
    
    def __init__(self, scale):
        self.s = (scale - 1) / (2*math.sqrt(2))
        self.t = 1/(2 * math.sqrt(2)*self.s)
        
    def values(self, x):
        return ((x[:,0]+self.t)**2 - (x[:,1]+self.t)**2).reshape(-1,1)*self.s
    def gradient(self, x):
        return (x+[self.t,self.t]) * [2,-2] *self.s

class NormOfGradient():
    def __init__(self, S):
        self.S = S

    def __call__(self, x):
        return np.sqrt(np.sum(self.S.gradient(x)**2, axis=1))

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

class PlaneWaveFromDirectionsRule(object):
    
    def __init__(self, S):
        self.S = S
        
    def populate(self, einfo):
        import pypwdg.core.bases.definitions as pcbb
        dir = self.S.gradient(einfo.origin)
        dir = dir / math.sqrt(np.sum(dir**2))    
        return [pcbb.PlaneWaves(dir,einfo.k)]

import pypwdg.parallel.main

from numpy import array,sqrt

k = 20
scale = 4.0
S = harmonic1(scale)

g = HarmonicDerived(k, S)

bdytag = "BDY"

#bnddata={bdytag:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
bnddata={bdytag:pcbd.dirichlet(g)}
entityton ={1:NormOfGradient(S)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([200,200])

def geterr(problem, basisrule):
    computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, 15)
    solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
#    mp.close()
#    pom.output2dsoln(bounds, solution, npoints)
#    pos.standardoutput(computation, solution, 20, bounds, npoints, None)
    return pos.comparetrue(bounds, npoints, g, solution)

ns = range(1,8)+range(8,16,2)+range(16,33,4)
pw1s = range(7,40,2)
pp1s = range(1,11)
pp2s = range(1,6)
pw2s = range(5,27,4)

pwerr = np.zeros((len(ns), len(pw1s)))
polyerr = np.zeros((len(ns), len(pp1s)))
polydirerr = np.zeros((len(ns), len(pp1s)))
polypwerr = np.zeros((len(ns), len(pp2s), len(pw2s)))

print "ns: ",ns
print "pw1s: ", pw1s
print "pp1s: ", pp1s
print "pw2s: ", pw2s
print "pp2s: ", pp2s

MAXDOF = 500000

for ni, n in enumerate(ns):
    mesh = tum.regularsquaremesh(n, bdytag)
    problem=psp.VariableNProblem(entityton, mesh,k, bnddata)
    
    for pi,p in enumerate(pw1s):
        if p * n * n * 2 > MAXDOF: break
        basisrule = pcbv.PlaneWaveVariableN(pcb.circleDirections(p))
        pwerr[ni,pi] = geterr(problem, basisrule)

    #    basisrule = pcbr.ReferenceBasisRule(pcbr.Dubiner(9))
        #basisrule = pcb.planeWaveBases(2, k, 20) 
        #basisrule = pcb.ProductBasisRule(pcbv.PlaneWaveVariableN(pcb.circleDirections(13)), pcbr.ReferenceBasisRule(pcbr.Dubiner(1)))
        
    #    pom.output2dfn(bounds, entityton[1], npoints)
    #    pom.output2dfn(bounds, g.values, npoints)
        
            
    for pi,p in enumerate(pp1s):   
        if p * (p+1) * n * n > MAXDOF: break
        basisrule = pcb.ProductBasisRule(PlaneWaveFromDirectionsRule(S), pcbr.ReferenceBasisRule(pcbr.Dubiner(p)))
        polydirerr[ni,pi] = geterr(problem, basisrule)
        basisrule = pcbr.ReferenceBasisRule(pcbr.Dubiner(p))
        polyerr[ni,pi] = geterr(problem, basisrule)

    for ppi, pp in enumerate(pp2s):
        for pwi, pw in enumerate(pw2s):
            break
            if pp * (pp+1) * n * n * pw > MAXDOF: break
            basisrule = pcb.ProductBasisRule(pcbv.PlaneWaveVariableN(pcb.circleDirections(pw)), pcbr.ReferenceBasisRule(pcbr.Dubiner(pp)))
            polypwerr[ni,ppi,pwi] = geterr(problem, basisrule)
    
    print "n = ", n
    print pwerr[ni]
    print polyerr[ni]
    print polydirerr[ni]
    print polypwerr[ni]
    print gc.collect()
    
#        mp.close()
#        pom.output2dsoln(bounds, solution, npoints)
#    pos.standardoutput(computation, solution, 20, bounds, npoints, None)

print "ns: ",ns
print "pw1s: ", pw1s
print "pp1s: ", pp1s
#print "pw2s: ", pw2s
#print "pp2s: ", pp2s

print pwerr
print polyerr
print polydirerr
#print polypwerr
