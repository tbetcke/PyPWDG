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
    print pwnorm2, gnorm2
    ip = np.dot(gv.conj().T,qw * pwv)
    proj = ip * ip.conj() / (gnorm2 * pwnorm2)
    ipd = np.dot(gv.conj().T, qw * pwv * 1j * k * np.dot(qx, ddirs))
    projd = 2 * np.real(ipd * ip.conj() / (gnorm2 * pwnorm2))
    ipdd = -np.dot(gv.conj().T, qw * k**2 * (np.dot(qx, -dirs) + np.abs(np.dot(qx, ddirs))**2))
    projdd = 2 * np.real((ipdd * ip.conj() + ipd * ipd.conj())/ (gnorm2 * pwnorm2))
    return theta, proj, projd, projdd


if __name__ == '__main__':
    N = 20
    k = 4    
    qxw = puq.squarequadrature(N)
    D = math.sqrt(2)
    D = 1
#    
#    mesh = tum.regularsquaremesh()
#    e = 0
#    mqs = pmmu.MeshQuadratures(mesh, puq.legendrequadrature(N))
#    qx = np.vstack([mqs.quadpoints(f) for f in mesh.etof[e]])
#    qw = np.concatenate([mqs.quadweights(f) for f in mesh.etof[e]])
#    qxw = (qx,qw)
    
    g = pcb.PlaneWaves(pcb.circleDirections(40)[15], k)
#    g = pcb.FourierHankel([-1,-0.5], [10], k)
#    g = pcb.BasisReduce(pcb.PlaneWaves(pcb.circleDirections(20)[[5,8]], k), [1,1])
#    g = pcb.BasisReduce(pcb.BasisCombine([pcb.FourierHankel([-1,-0.5], [0], k), pcb.FourierHankel([-0.2,0.5], [0], k)]), [1,1])
#    g = pcb.FourierBessel([0.25,0.25],[20], k)

    theta, proj, projd, projdd = pwprod(g, qxw, k, 500)
#    print errs[0]
    mp.plot(theta, proj[0])
    mp.plot(theta, projd[0])
    mp.plot(theta, projdd[0])
    mp.plot(theta, D**2 * k * np.sqrt(proj[0] / 3))
    mp.show()