'''
Created on Mar 22, 2011

@author: joel
'''
import numpy as np
import math
import pypwdg.utils.optimisation as puo
import scipy.linalg as sl

class BoundaryDataFit(object):
    def __init__(self, bc, normal, quadrule):
        self.qp, qw = quadrule
        self.qwsqrt = np.sqrt(qw).ravel()
        rc = bc.r_coeffs
        self.urp = (bc.values(self.qp) * rc[0] + bc.derivs(self.qp, normal) * rc[1]).ravel() * self.qwsqrt
        self.l2urp = math.sqrt(np.vdot(self.urp,self.urp))
        self.lc = bc.l_coeffs
        self.normal = normal
    
    def optimise(self, basis):
        basisvals = (basis.values(self.qp)*self.lc[0] + basis.derivs(self.qp, self.normal) * self.lc[1]) * self.qwsqrt.reshape(-1,1)
        coeffs = sl.lstsq(basisvals, self.urp)[0]
        err = np.abs(np.dot(basisvals, coeffs).ravel() - self.urp)/self.l2urp
        return coeffs, err

def initialrt(mesh, bdydata, k, mqs, maxspace):
    """ Find starting points and directions for ray-tracing """
    
    ftoparams = {}
    for (bdy, bc) in bdydata.items():
        faces = mesh.entityfaces[bdy]            
        for f in faces.tocsr().indices:
            gen, ini = puo.pwbasisgeneration(k, 1)
            qp = mqs.quadpoints(f)
            qw = mqs.quadweights(f)
            linearopt = BoundaryDataFit(bc, mesh.normals[f], (qp,qw))
            params = puo.optimalbasis3(linearopt.optimise, gen, ini, None, lambda params:params)
            print params
            ftoparams[f] = params

            