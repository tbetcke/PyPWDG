'''
Created on 1 Nov 2010

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import pypwdg.parallel.distributeddict as ppdd
import pypwdg.core.bases as pcb
import pypwdg.setup as ps
import pypwdg.utils.optimisation as puo
import pypwdg.utils.optx as puoptx

import numpy as np

def augmentparams(params, npw, dim):
    if params==None: params = np.zeros((0,dim))
    if len(params) < npw:
        params = np.vstack((params, pcb.circleDirections(npw - len(params))))
    return normalise(params, npw)

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
        if npw > 0: self.params = augmentparams(params, npw, len(origin))
    
    def pwbasis(self, params):
        return pcb.PlaneWaves(normalise(params, self.npw), self.k) if self.npw > 0 else None
    
    def newparams(self, params):
        return PWFBCreator(self.k,self.origin,self.npw,self.nfb,params)
     
    def getBasis(self):        
        fbbasis = pcb.FourierBessel(self.origin, np.arange(0,self.nfb) - self.nfb/2, self.k)
        if self.npw > 0:
            pwbasis = self.pwbasis(self.params)
            return [pwbasis, fbbasis]
        else:
            return [fbbasis]

    def nearbybases(self, x):
        nearby = [self]
#        return nearby
        k,origin,npw,nfb, params = self.k,self.origin,self.npw,self.nfb, self.params
        nearby.append(PWFBCreator(k,origin,npw,nfb+1,params))
        if nfb > 3: nearby.append(PWFBCreator(k,origin,npw,nfb-1,params))
        nearby.append(PWFBCreator(k,origin,npw+1,nfb,params))
        xpw = x[:len(params)]
        if npw > 0: nearby.append(PWFBCreator(k,origin,npw-1,nfb,params[np.abs(xpw) != np.min(np.abs(xpw))]))
        return nearby
       
def origin(mesh, e):
    return np.average(mesh.nodes[mesh.elements[e]], axis = 0)

class InitialPWFBCreator(object):
    def __init__(self, mesh, k, npw, nfb):
        self.mesh = mesh
        self.k = k
        self.npw = npw
        self.nfb = nfb
    
    def __call__(self, e):
        return PWFBCreator(self.k, origin(self.mesh, e), self.npw, self.nfb)

@ppd.distribute()
class BasisController(object):
    
    def __init__(self, mesh, mqs, etob, ibc):
        self.mqs = mqs
        self.mesh = mesh
        self.etobc = dict([(e, ibc(e)) for e in mesh.partition])
        self.etonbcs = {}
        self.populate()
    
    @ppd.parallelmethod(None, None)
    def populate(self):
        for (e, bc) in self.etobc.iteritems():
            self.etob[e] = bc.getBasis()
    
    @ppd.parallelmethod(None, None)
    def selectNearbyBasis(self, etonbc):
        for e in self.etobc.keys():
            self.etobc[e] = self.etonbcs[e][etonbc[e]]
        self.etonbcs = {}
        self.populate()
    
    @ppd.parallelmethod(None, ppdd.combinedict)    
    def getNearbyBases(self, indices, x):
        for e in self.mesh.partition:
            xe = x[indices[e]:indices[e+1]]
            bc = self.etopwfbc[e]
            fs = self.mesh.etof[e]
            qp = np.vstack([self.mqs.quadpoints(f) for f in fs])
            qw = np.concatenate([self.mqs.quadweights(f) for f in fs])
            lsf = puo.LeastSquaresFit(pcb.BasisReduce(pcb.BasisCombine(bc.getBasis()),xe).values, (qp,qw))            
            nbcs = bc.nearbybases(xe)
            optimisednbcs = []
            for i, nbc in enumerate(nbcs):
                newnbc = puo.optimalbasis3(lsf.optimise, nbc.pwbasis, nbc.params, None, nbc.newparams) if nbc.npw > 0 else nbc
                (_, l2err) = lsf.optimise(pcb.BasisCombine(newnbc.getBasis()))
                optimisednbcs.append((i, newnbc.n, sum(l2err)))
            self.etonbcs[e] = optimisednbcs
        return self.etonbcs 
            
    

class AdaptiveComputation(object):
    
    def __init__(self, problem, ibc, factor = 1):
        self.problem = problem
        self.manager = ppdd.ddictmanager(ppdd.elementddictinfo(problem.mesh), True)
        self.controller = BasisController(problem.mesh, problem.mqs, self.manager.getDict(), ibc)
        self.manager.sync()   
        self.factor = factor
        self.nelements = problem.mesh.nelements
    
    def solve(self):
        self.EtoB = pcb.ElementToBases(self.etob, self.problem.mesh)
        self.solution = ps.Computation(self.problem, self.EtoB, False).solve()
        return self.solution
    
    def adapt(self):
        etonbcs = self.ab.evaluateNearbyBases(self.EtoB.indices, self.solution.x)
        oldn = self.EtoB.indices[-1]
        gain = np.ones((self.nelements, 3)) * -1
        gain[:,0] = 0
        bestbcs = np.empty((self.nelements, 3),dtype=int)
        totaldof = 0
        for e in range(self.nelements):
            nbcs = sorted(etonbcs[e], lambda (i1, nbc1,err1), (i2, nbc2, err2):int(np.sign(err2 - err1)) if nbc1.n==nbc2.n else nbc1.n - nbc2.n)
            print [(i, nbc.n,err) for (i, nbc,err) in nbcs] 
            (i0, nbc0, err0) = nbcs[0]
            
            for (i, nbc, err) in nbcs[1:]:
                ddof = nbc.n - nbc0.n
                if ddof==0: (i0, nbc0, err0) = i, nbc, err
                else:
                    bestbcs[e, ddof] = i
                    gain[e,ddof] = err0 - err
            totaldof+=nbc0.n                           
            bestbcs[e,0] = i0         
        
        print gain.transpose()
        doftospend = int(oldn * self.factor) - totaldof
        print doftospend
        if doftospend:      
            x,errest = puoptx.optx(gain.transpose(), doftospend)
            print x
            newbcs = [ebcs[xi] for ebcs, xi in zip(bestbcs, x)]
        else: newbcs = bestbcs[:,0]
        self.ab.selectNearbyBasis(newbcs)
        
        