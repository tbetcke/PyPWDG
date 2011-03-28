'''
Created on Mar 28, 2011

@author: joel
'''
import numpy as np
import pypwdg.core.bases as pcb
import pypwdg.utils.quadrature as puq
import test.utils.mesh as tum
import pypwdg.mesh.meshutils as pmmu
import math
import matplotlib.pyplot as mp

def pwprod(g, qxw, k, n):
    qx, qw = qxw
    qw = qw.reshape(-1,1)
    pw = pcb.PlaneWaves(pcb.circleDirections(n), k)
    pwv = pw.values(qx)
    gv = g.values(qx).reshape(-1,1)
    pwnorm2 = sum(qw)
    gnorm2 = np.dot(gv.conj().T , qw * gv)
#    print qx.shape, gv.shape, pwv.shape, qw.shape, gnorm2.shape
#    return  np.sqrt(1-np.dot(gv.conj().T, qw * pwv)**2 / (gnorm2 * pwnorm2))
    print gnorm2 
    return  np.sqrt(np.dot(gv.conj().T,qw * pwv)**2 / (gnorm2 * pwnorm2))


if __name__ == '__main__':
    N = 20
    k = 10    
    qxw = puq.trianglequadrature(N)
    
    mesh = tum.regularsquaremesh()
    e = 0
    mqs = pmmu.MeshQuadratures(mesh, puq.legendrequadrature(N))
    qx = np.vstack([mqs.quadpoints(f) for f in mesh.etof[e]])
    qw = np.concatenate([mqs.quadweights(f) for f in mesh.etof[e]])
    qxw = (qx,qw)
    
    g = pcb.PlaneWaves(pcb.circleDirections(20)[5], k)
#    g = pcb.FourierHankel([-2,-2], [0], k)
#    g = pcb.BasisReduce(pcb.PlaneWaves(pcb.circleDirections(20)[[5,8]], k), [1,1])

    errs = pwprod(g, qxw, k, 500)
#    print errs[0]
    mp.plot(errs[0])
    mp.show()