'''
Created on Mar 27, 2011

@author: joel
'''
import test.utils.mesh as tum
import pypwdg.mesh.meshutils as pmmu
import pypwdg.utils.quadrature as puq
import pypwdg.core.bases as pcb
import numpy as np
import scipy.linalg as sl
import math
import matplotlib.pyplot as mp

def geterrs(g, (qx,qw), auxbasis, k, n):
    sqw = np.sqrt(qw).reshape(-1,1)
    auxv = auxbasis.values(qx) * sqw
    print auxv.shape
    pws = pcb.PlaneWaves(pcb.circleDirections(n),k) 
    pwv = pws.values(qx) * sqw
    gv = g.values(qx)* sqw
    errs = np.empty(n)
    for i in range(n):
        bv = np.hstack((auxv, pwv[:,i].reshape(-1,1)))
        coeffs = sl.lstsq(bv, gv)[0]
        resids = np.dot(bv, coeffs) - gv
        
        errs[i] = math.sqrt(np.vdot(resids, resids))
    return errs  



if __name__ == '__main__':
    k = 10
    N = 40
    FB = -1
#    qxw = puq.trianglequadrature(N)
#    origin = [0.25,0.25]

    mesh = tum.regularsquaremesh()
    mqs = pmmu.MeshQuadratures(mesh, puq.legendrequadrature(N))

    e = 0
    qx = np.vstack([mqs.quadpoints(f) for f in mesh.etof[e]])
    qw = np.concatenate([mqs.quadweights(f) for f in mesh.etof[e]])
    qxw = (qx,qw)
    origin = pmmu.elementcentres(mesh)[e]

#    e = 0
#    f = 0
#    qx = mqs.quadpoints(mesh.etof[e][f])
#    qw = mqs.quadweights(mesh.etof[e][f])
#    qxw = (qx,qw)
#    origin = pmmu.elementcentres(mesh)[e]
    
    
#    g = pcb.PlaneWaves(np.array([[3.0/5, 4.0/5]]),k)
    g = pcb.BasisReduce(pcb.PlaneWaves(pcb.circleDirections(50)[[12,18,40],:],k), [1,1,1])
#    g = pcb.FourierHankel([-2,-2], [0], k)
    
    auxbasis = pcb.FourierBessel(origin, range(-FB, FB+1), k)
    errs = geterrs(g,qxw, auxbasis, k, 500)
    mp.plot(errs)
    mp.show() 
    