'''
Created on Mar 28, 2011

@author: joel
'''
import numpy as np
import pypwdg.core.bases as pcb
import pypwdg.utils.quadrature as puq
import pypwdg.test.utils.mesh as tum
import pypwdg.mesh.meshutils as pmmu
import math
import matplotlib.pyplot as mp
import pypwdg.utils.timing as put
import scipy.linalg as sl
import pypwdg.adaptivity.planewave as pap

def project(g, qxw, k, theta):
    dir = np.array([[math.cos(theta), math.sin(theta)]])
    qx, qw = qxw    
    pw = pcb.PlaneWaves(dir, k)
    pwv = pw.values(qx)
    gv = g.values(qx).reshape(-1,1) # * (1+qx[:,0]*2 + qx[:,1]).reshape(-1,1)
    gvqw = gv * qw.reshape(-1,1)    
    pwnorm = math.sqrt(sum(qw))
    gnorm = math.sqrt(np.abs(np.dot(gvqw.conj().T,  gv)))    
        
    ipnormed = np.dot(gvqw.conj().T,pwv) / (pwnorm)
    return pcb.BasisReduce(pcb.BasisCombine([g, pw]), np.array([1.0, -ipnormed[0,0]])) 


def pwproduniform(g, qxw, k, n):
    theta = np.linspace(0, 2*math.pi, n, endpoint=False)
    return (theta,)+ pap.L2Prod(g, qxw, k).products(theta)

def qrp(qxw, dirs, k):
    qx,qw = qxw
    P = np.sqrt(qw.reshape(-1,1)) * pcb.PlaneWaves(dirs, k).values(qx)
    Q,R = sl.qr(P)
    print R

if __name__ == '__main__':
    N = 20
    k = 20    
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
    
    qrp(qxw, pcb.circleDirections(4), k)
    
    g = pcb.PlaneWaves(pcb.circleDirections(40)[15], k)
    g = pcb.FourierHankel([-1,-0.5], [10], k)
    g = pcb.BasisReduce(pcb.PlaneWaves(pcb.circleDirections(20)[[5,8]], k), [3,1])
    g = pcb.BasisReduce(pcb.BasisCombine([pcb.FourierHankel([-1,-0.5], [0], k), pcb.FourierHankel([-0.2,0.5], [0], k)]), [1,1])
#    g = pcb.FourierBessel([0.25,0.25],[20], k)

    t1 = pap.findpw(pap.L2Prod(g.values, qxw, k), D, maxtheta = 1)
    g1 = project(g, qxw, k, t1)
    t2 = pap.findpw(pap.L2Prod(g1.values, qxw, k), D, maxtheta = 1)
    g2 = project(g1, qxw, k, t2)
    print t1
    print t2

    theta, proj, projd, projdd = pwproduniform(g.values, qxw, k, 500)
#    print errs[0]
    mp.plot(theta, proj[0])
#    mp.plot(theta, projd[0])
#    mp.plot(theta, projdd[0])
#    mp.plot(theta, (np.roll(projd[0],-1) - projd[0]) / (np.roll(theta,-1) - theta))
#    mp.plot(theta, D**2 * k * np.sqrt(proj[0] / 3))
    mp.show()