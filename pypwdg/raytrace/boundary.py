'''
Created on Mar 22, 2011

@author: joel
'''
import numpy as np
import math
import pypwdg.utils.optimisation as puo
import pypwdg.core.bases as pcb
import scipy.linalg as sl

class BoundaryDataFit(object):
    def __init__(self, bc, normal, quadrule):
        self.qp, qw = quadrule
        self.qwsqrt = np.sqrt(qw).ravel()
        rc = bc.r_coeffs
        self.urp = (bc.values(self.qp) * rc[0] + bc.derivs(self.qp, normal) * rc[1]).ravel() * self.qwsqrt
#        print self.urp
        self.l2urp = math.sqrt(np.vdot(self.urp,self.urp))
        self.lc = bc.l_coeffs
        self.normal = normal
#        print self.lc
#        print self.qwsqrt
    
    def optimise(self, basis):
#        print basis.values(self.qp).shape, self.qwsqrt.shape
        basisvals = ((basis.values(self.qp)*self.lc[0] + basis.derivs(self.qp, self.normal) * self.lc[1]) * self.qwsqrt.reshape(-1,1))
#        print basisvals
        coeffs = sl.lstsq(basisvals, self.urp)[0]
        err = np.abs(np.dot(basisvals, coeffs).ravel() - self.urp)/self.l2urp
#        print coeffs, err
        return coeffs, err


def initialrt(mesh, bdydata, k, mqs, maxspace):
    """ Find starting points and directions for ray-tracing """
    def normalise(params):
        p = params.reshape(-1, mesh.dim)
        return p / np.sqrt(np.sum(p**2, axis=1)).reshape(-1,1)
    
    ftoparams = {}
    for (bdy, bc) in bdydata.items():
        faces = mesh.entityfaces[bdy]            
        for f in faces.tocsr().indices:
            gen = lambda params: pcb.PlaneWaves(normalise(params), k)
            ini = pcb.circleDirections(4)
            print ini.shape
            qp = mqs.quadpoints(f)
            qw = mqs.quadweights(f)
            bdf = BoundaryDataFit(bc, mesh.normals[f], (qp,qw))
            params = puo.optimalbasis3(bdf.optimise, gen, ini, None, normalise)
            print params
            print bdf.optimise(gen(params))[0]
            ftoparams[f] = params

            