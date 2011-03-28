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
import pypwdg.utils.timing as put

@put.print_timing
def pwprod(g, qxw, k, n):
    theta = np.linspace(0, 2*math.pi, n, endpoint=False)
    dirs = np.vstack([np.cos(theta), np.sin(theta)])
    ddirs = np.vstack([-dirs[1], dirs[0]])
    qx, qw = qxw
    qw = qw.reshape(-1,1)
    pw = pcb.PlaneWaves(dirs.T, k)
    pwv = pw.values(qx)
    gv = g.values(qx).reshape(-1,1) # * (1+qx[:,0]*2 + qx[:,1]).reshape(-1,1)
    pwnorm2 = sum(qw)
    gnorm2 = np.dot(gv.conj().T , qw * gv)
    ip = np.dot(gv.conj().T,qw * pwv)
    proj = ip * ip.conj() / (gnorm2 * pwnorm2)
    print np.dot(qx, dirs).shape, pwv.shape
    ipd = np.dot(gv.conj().T, qw * pwv * 1j * k * np.dot(qx, ddirs))
    projd = 2 * np.real(ipd * ip.conj() / (gnorm2 * pwnorm2))
    return proj, projd


if __name__ == '__main__':
    N = 20
    k = 10    
    qxw = puq.trianglequadrature(N)
#    
#    mesh = tum.regularsquaremesh()
#    e = 0
#    mqs = pmmu.MeshQuadratures(mesh, puq.legendrequadrature(N))
#    qx = np.vstack([mqs.quadpoints(f) for f in mesh.etof[e]])
#    qw = np.concatenate([mqs.quadweights(f) for f in mesh.etof[e]])
#    qxw = (qx,qw)
    
#    g = pcb.PlaneWaves(pcb.circleDirections(20)[5], k)
    g = pcb.FourierHankel([-1,-0.5], [10], k)
#    g = pcb.BasisReduce(pcb.PlaneWaves(pcb.circleDirections(20)[[5,8]], k), [1,1])
    g = pcb.BasisReduce(pcb.BasisCombine([pcb.FourierHankel([-1,-0.5], [0], k), pcb.FourierHankel([-0.2,0.5], [0], k)]), [1,1])

    proj, projd = pwprod(g, qxw, k, 500)
#    print errs[0]
    mp.plot(proj[0])
    mp.plot(projd[0])
    mp.show()