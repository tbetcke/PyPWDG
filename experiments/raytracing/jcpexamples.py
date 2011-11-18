'''
Created on Nov 18, 2011

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
import time
import random

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
    
    def __init__(self, S, err = 0):
        self.S = S
        self.err = err
        
    def populate(self, einfo):
        import pypwdg.core.bases.definitions as pcbb
        dir = self.S.gradient(einfo.origin)
        M = np.eye(2) + np.array([[0,1],[-1,0]]) * self.err * random.random()
        dir = np.dot(M, dir)
        dir = dir / math.sqrt(np.sum(dir**2))    
        return [pcbb.PlaneWaves(dir,einfo.k)]

def variableNhConvergence(Ns, nfn, bdycond, basisrule, process, k = 20, scale = 4.0):
    #bnddata={bdytag:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
    bdytag = "BDY"    
    bnddata={bdytag:bdycond}
    entityton ={1:nfn}
    for n in Ns:
        mesh = tum.regularsquaremesh(n, bdytag)
        problem=psp.VariableNProblem(entityton, mesh,k, bnddata)
        computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, 15)
        solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
        process(n, solution)
        
class FileOutput():
    
    def __init__(self, name, header, g, bounds, npoints):
        self.ftxt = open(name + ".txt", 'a')
        self.ftxt.write(header+'\n')
        self.bounds = bounds
        self.npoints = npoints
        self.g = g
    
    def process(self, n, solution):
        err = pos.comparetrue(self.bounds, self.npoints, self.g, solution)
        print n, err
        self.ftxt.write("%s, "%(err))
    

def analytichconvergence(maxN, k = 20, scale = 4.0):    
    fileroot = "hconv.k%s.scale%s"%(k,scale)
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([k * scale * 10,k * scale * 10], dtype=int)
    S = harmonic1(scale)
    g = HarmonicDerived(k, S)   
    nfn = NormOfGradient(S)
    bdycond = pcbd.dirichlet(g)
    
    npw = 15
    pdeg = 2
    Ns = range(1,maxN+1)
    
    pw = pcbv.PlaneWaveVariableN(pcb.circleDirections(npw))
    fo = FileOutput(fileroot + 'uniformpw%s'%npw, str(Ns), g, bounds, npoints)
    variableNhConvergence(Ns, nfn, bdycond, pw, fo.process, k, scale)
    
    poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(pdeg))
    fo = FileOutput(fileroot + 'poly%s'%pdeg, str(Ns), g, bounds, npoints)
    variableNhConvergence(Ns, nfn, bdycond, poly, fo.process, k, scale)

    for err in [0, 0.02, 0.2]:
        rt = PlaneWaveFromDirectionsRule(S, err)
        fo = FileOutput(fileroot + 'rt-err%s'%err, str(Ns), g, bounds, npoints)
        variableNhConvergence(Ns, nfn, bdycond, rt, fo.process, k, scale)
        for p in [1,2,3,4]:
            poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(p))
            polyrt = pcb.ProductBasisRule(poly, rt)
            fo = FileOutput(fileroot + 'poly%srt-err%s'%(p,err), str(Ns), g, bounds, npoints)
            variableNhConvergence(Ns, nfn, bdycond, polyrt, fo.process, k, scale)
        


import pypwdg.parallel.main

if __name__ == '__main__':
    analytichconvergence(30)
    
    
    
    