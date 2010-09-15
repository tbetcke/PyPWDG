import numpy as np
import scipy.optimize as so
import scipy.linalg as sl
import math
import pypwdg.core.bases as pcb
import pypwdg.mesh.meshutils as pmmu

import pypwdg.parallel.decorate as ppd
    
def optimalbasis(u, basisgenerator, initialparams, quadrule, retcoeffs = False):
    qp, qw = quadrule
    qwsqrt = np.sqrt(qw).flatten()
    urp = u(qp).flatten() * qwsqrt
    def lstsqerr(params, ret=False):
#        print "params ", params
        basis = basisgenerator(params)
        basisvals = basis.values(qp) * qwsqrt.reshape(-1,1)
        coeffs = sl.lstsq(basisvals, urp)[0]
        err = np.dot(basisvals, coeffs) - urp
        l2err = math.sqrt(np.vdot(err,err))
        if ret: return basis, coeffs, l2err
        else: return l2err         

    optparams = so.fmin_powell(lstsqerr, initialparams, ftol = 1e-2, disp=False)
    
    if retcoeffs:
        return lstsqerr(optparams, True)
    else:
        return basisgenerator(optparams) 

def pwbasisgeneration(k, npw):
    initialtheta = np.arange(npw).reshape((-1,1)) * 2*math.pi / npw
    generator = lambda theta: pcb.PlaneWaves(np.hstack((np.cos(theta.reshape(-1,1)), np.sin(theta.reshape(-1,1)))), k)
    return generator, initialtheta

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

@ppd.parallel(lambda n : lambda m, bs, X, q, k, npw, o: [((m,bb,X[i0:i1],q,k,npw, o+eo),{}) for bb, (i0,i1), eo in splitbasis(n, bs)])
def generatepwbasis(mesh, oldbasis, X, refquad, k, npws, eltoffset):
    gen, ini = pwbasisgeneration(k, npws)
    return generatebasis(mesh, oldbasis, X, gen, ini, refquad, eltoffset)
