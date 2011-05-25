'''
Created on Jan 1, 2011

@author: joel
'''

import numpy as np
import scipy.optimize as so
import scipy.linalg as sl
import math
import pypwdg.core.bases as pcb
import pypwdg.mesh.meshutils as pmmu
import pypwdg.utils.timing as put
import pypwdg.parallel.decorate as ppd

class PWPenaltyBasisGenerator(object):
    def __init__(self, k, alpha, dim):
        self.k = k
        self.alpha = alpha
        self.dim = dim
    
    def genbasis(self, params):
#        print params
        return pcb.PlaneWaves(params.reshape(-1,self.dim), self.k)
    
    def penalty(self, params):        
        p = params.reshape(-1, self.dim)        
        pen = self.alpha * np.log(np.sum(p**2, axis=1))
        return pen
    
    def finalbasis(self, params):
        p = params.reshape(-1, self.dim)
        return self.genbasis(p / np.sqrt(np.sum(p**2, axis=1)).reshape(-1,1)) 

class LeastSquaresFit(object):
    def __init__(self, u, quadrule):
        self.qp, qw = quadrule
        self.qwsqrt = np.sqrt(qw).ravel()
        self.urp = u(self.qp).ravel() * self.qwsqrt
        self.l2urp = math.sqrt(np.abs(np.vdot(self.urp,self.urp)))
    
    def optimise(self, basis):
        basisvals = basis.values(self.qp) * self.qwsqrt.reshape(-1,1)
        coeffs = sl.lstsq(basisvals, self.urp)[0]
        err = np.abs(np.dot(basisvals, coeffs).ravel() - self.urp)/self.l2urp
        return coeffs, err

#@put.print_timing    
def optimalbasis3(linearopt, basisgenerator, iniparams, penalty = None, finalobject = None):
    if finalobject is None: finalobject = basisgenerator
    def nonlinearfn(params):
#        print params
        basis = basisgenerator(params)
        err = linearopt(basis)[1]
        return err if penalty is None else np.concatenate((err, penalty(params)))
    
    optparams, a, b, msg,ier  = so.leastsq(nonlinearfn, iniparams.flatten(), full_output = True)     
#    print msg, ier
    return finalobject(optparams)
    
@put.print_timing    
def optimalbasis2(u, basisgenerator, initialparams, quadrule, extrabasis = None):
    qp, qw = quadrule
    qwsqrt = np.sqrt(qw).flatten()
    urp = u(qp).flatten() * qwsqrt
    l2urp = math.sqrt(np.vdot(urp,urp))

    def linearopt(basis):
        basisvals = basis.values(qp) * qwsqrt.reshape(-1,1)
        coeffs = sl.lstsq(basisvals, urp)[0]
        err = np.abs(np.dot(basisvals, coeffs).flatten() - urp)/l2urp
        return coeffs, err
    
    def nonlinearfn(params):
#        print params
        basis = basisgenerator(params)
        coeffs, err = linearopt(basis)
        return err
    
    optparams = so.leastsq(nonlinearfn, initialparams.flatten())[0]
    pwbasis = basisgenerator(optparams)
    basis = pcb.BasisCombine([pwbasis, extrabasis]) if extrabasis is not None else pwbasis
    return basis, linearopt(basis)

@put.print_timing    
def optimalbasis(u, basisgenerator, initialparams, quadrule, retcoeffs = False):
    qp, qw = quadrule
    qwsqrt = np.sqrt(qw).flatten()
    urp = u(qp).flatten() * qwsqrt
    l2urp = math.sqrt(np.vdot(urp,urp))
    
    def lstsqerr(params, ret=False):
#        print "params ", params
        basis = basisgenerator(params)
        basisvals = basis.values(qp) * qwsqrt.reshape(-1,1)
        coeffs = sl.lstsq(basisvals, urp)[0]
        err = np.dot(basisvals, coeffs) - urp
        l2err = math.sqrt(np.vdot(err,err))/l2urp
        if ret: return basis, coeffs, l2err
        else: return l2err         

    optparams = so.fmin_powell(lstsqerr, initialparams, ftol = 1e-4, disp=False)
    
    if retcoeffs:
        return lstsqerr(optparams, True)
    else:
        return basisgenerator(optparams) 

def pwbasisgeneration(k, npw):
    initialtheta = np.arange(npw).reshape((-1,1)) * 2*math.pi / npw
    generator = lambda theta: pcb.PlaneWaves(np.hstack((np.cos(theta.reshape(-1,1)), np.sin(theta.reshape(-1,1)))), k)
    return generator, initialtheta

def pwfbbasisgeneration(k, origin, npw, nfb):
    initialtheta = np.arange(npw).reshape((-1,1)) * 2*math.pi / npw
    def gen(theta):
        basis = pcb.PlaneWaves(np.hstack((np.cos(theta.reshape(-1,1)), np.sin(theta.reshape(-1,1)))), k)
        if nfb >= 0:
            fb = pcb.FourierBessel(origin, np.arange(-nfb, nfb+1), k)
            basis = pcb.BasisCombine([basis, fb])
        return basis
    return gen, initialtheta
            

def generatebasis(mesh, oldbasis, x, generator, initialparams, refquad, eltoffset):
    indx = 0 # keep track of what we're indexing in x
    newbasis = []
    meq = pmmu.MeshElementQuadratures(mesh, refquad)
    for e, bs in enumerate(oldbasis): # Iterate over all the elements
        # Determine the function that we're trying to approximate on this element
        newindx = indx + sum([b.n for b in bs])
        def u(points):
            v = np.hstack([b.values(points) for b in bs])
            return np.dot(v, x[indx:newindx])
        bnew = optimalbasis(u, generator, initialparams, (meq.quadpoints(e+eltoffset), meq.quadweights(e+eltoffset)))
        newbasis.append([bnew])
        indx = newindx        
#    for e,b in enumerate(newbasis): print e, b[0].directions
    return newbasis

# this is a bit of a mess.  We need a distributed adaptivity class
def splitbasis(n, bs):
    bspl = ppd.partitionlist(n, bs)
    idx = np.cumsum([0] + [sum([b.n for b in sum(bsp,[])]) for bsp in bspl])
    eltoffsets = np.cumsum([0] + map(len, bspl[:-1]))
    return zip(bspl, zip(idx[:-1],idx[1:]), eltoffsets)

def generatepwbasis(mesh, oldbasis, X, refquad, k, npws, eltoffset):
    gen, ini = pwbasisgeneration(k, npws)
    return generatebasis(mesh, oldbasis, X, gen, ini, refquad, eltoffset)

