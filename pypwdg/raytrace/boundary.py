'''
Created on Mar 22, 2011

@author: joel
'''

import pypwdg.utils.optimisation as puo

class BoundaryDataFit(object):
    def __init__(self, bc, normal, quadrule):
        self.qp, qw = quadrule
        self.qwsqrt = np.sqrt(qw).ravel()
        rc = bc.r_coeffs
        self.urp = (bc.values(self.qp) * rc[0] + bc.derivs(self.qp) * rc[1]).ravel() * self.qwsqrt
        self.l2urp = math.sqrt(np.vdot(self.urp,self.urp))
    
    def optimise(self, basis):
        basisvals = basis.values(self.qp) * self.qwsqrt.reshape(-1,1)
        coeffs = sl.lstsq(basisvals, self.urp)[0]
        err = np.abs(np.dot(basisvals, coeffs).ravel() - self.urp)/self.l2urp
        return coeffs, err

def initialrt(mesh, bdydata, k, mqs, maxspace):
    """ Find starting points and directions for ray-tracing """
    
    for (bdy, bc) in bdydata.items():
        faces = mesh.entityfaces[bdy]            
        for f in faces.tocsr().indices:
            gen, ini = puo.pwbasisgeneration(k, 1)
            x = mqs.quadpoints(f)
            w = mqs.quadweights(f)
            linearopt = puo.LeastSquaresFit()

            