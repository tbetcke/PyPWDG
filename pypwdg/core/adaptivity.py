'''
Created on 1 Nov 2010

@author: joel
'''

import pypwdg.core.bases as pcb
import pypwdg.setup as ps
import pypwdg.utils.optimisation as puo
import pypwdg.utils.optx as puoptx

import math
import numpy as np

def augmentparams(params, npw, dim):
    if params==None: params = np.zeros((0,dim))
    if len(params) < npw:
        params = np.vstack((params, pcb.circleDirections(npw - len(params))))
    return params

def normalise(params, npw):
    p = params.reshape(npw, -1)
    return p / np.sqrt(np.sum(p**2, axis=1)).reshape(npw,1)        

class PWFBCreator(object):
    def __init__(self, k, origin, npw, nfb, params = None):        
        self.k = k
        self.origin = origin
        self.npw = npw
        self.nfb = nfb
        self.n = npw + nfb        
        self.params = augmentparams(params, npw, len(origin))
    
    def pwbasis(self, params):
        return pcb.PlaneWaves(normalise(params, self.npw), self.k)
    
    def newparams(self, params):
        return PWFBCreator(self.k,self.origin,self.npw,self.nfb,params)
     
    def getBasis(self):
        pwbasis = self.pwbasis(self.params)
        fbbasis = pcb.FourierBessel(self.origin, np.arange(0,self.nfb) - self.nfb/2, self.k)
        return [pwbasis, fbbasis]

    def nearbybases(self, x):
        nearby = [self]
        k,origin,npw,nfb, params = self.k,self.origin,self.npw,self.nfb, self.params
        nearby.append(PWFBCreator(k,origin,npw,nfb+1,params))
        if nfb > 3: nearby.append(PWFBCreator(k,origin,npw,nfb-1,params))
        nearby.append(PWFBCreator(k,origin,npw+1,nfb,params))
        if npw > 0: nearby.append(PWFBCreator(k,origin,npw,nfb,params[np.abs(x) != np.min(np.abs(x))]))
        return nearby

def origin(mesh, e):
    return np.average(mesh.nodes[mesh.elements[e]], axis = 0)

def newAdaptiveBasis(mesh, k, npw, nfb, mqs):
    etopwfbc = dict([(e, PWFBCreator(k, origin(mesh, e), npw, nfb)) for e in range(mesh.nelements)])
    return AdaptiveBasis(mesh, mqs, etopwfbc)

class AdaptiveBasis(object):
    
    def __init__(self, mesh, mqs, etopwfbc):
        self.etopwfbc = etopwfbc
        self.mqs = mqs
        self.mesh = mesh
    
    def getBases(self):
        return dict([(e, bc.getBasis()) for (e, bc) in self.etopwfbc.iteritems()])
    
    def newBCs(self, etopwfbc):
        return AdaptiveBasis(self.mesh, self.mqs, etopwfbc)
    
    def evaluateNearbyBases(self, indices, x):
        etonbcs = {}
        for e in self.mesh.partition:
            bc = self.etopwfbc[e]
            fs = self.mesh.etof[e]
            qp = np.vstack([self.mqs.quadpoints(f) for f in fs])
            qw = np.concatenate([self.mqs.quadweights(f) for f in fs])
            lsf = puo.LeastSquaresFit(pcb.BasisCombine(bc.getBasis()).values, (qp,qw))            
            nbcs = bc.nearbybases()
            nearbybases = []
            for nbc in nbcs:
                newnbc, (_, l2err) = puo.optimalbasis3(lsf.optimise, nbc.pwbasis, nbc.params, nbc.newparams)
                nearbybases.append((newnbc, l2err))
            etonbcs[e] = nbcs
        return etonbcs 
            
    

class AdaptiveComputation(object):
    
    def __init__(self, problem, initialpw, initialfb, factor = 1):
        self.problem = problem
        self.ab = newAdaptiveBasis(problem.mesh, problem.k, initialpw, initialfb, problem.mqs)
        self.factor = factor
        self.nelements = problem.mesh.nelements
    
    
    def solve(self):
        self.etob = pcb.ElementToBases(self.problem.mesh)
        self.etob.setEtoB(self.ab.getBases()) # change this once etob is immutable
        self.solution = ps.Computation(self.problem, self.etob, False).solve()
        return self.solution
    
    def adapt(self):
        etonbcs = self.ab.evaluateNearbyBases(self.etob.indices, self.solution.x)
        oldn = self.etob.indices[-1]
        gain = np.zeros(self.nelements, 3)
        bestbcs = np.empty((self.nelements, 3),dtype=object)
        for e in range(self.nelements):
            nbcs = sorted(etonbcs[e], lambda (nbc,err):nbc.n * 10 + math.atan(err))
            (nbc0, err0) = nbcs[0]
            
            for (nbc, err) in nbcs[1:]:
                ddof = nbc.n = nbc0.n
                if ddof==0: (nbc0, err0) = nbc, err
                else:
                    bestbcs[e, ddof] = nbc
                    gain[e,ddof] = err - err0
                                        
            bestbcs[e,0] = nbc0         
        
        x,errest = puoptx.optx(gain.transpose, int(oldn * self.factor))
        self.ab = self.ab.newBCs(dict(zip(np.arange(self.nelements), bestbcs[x])))
        print x
        
        