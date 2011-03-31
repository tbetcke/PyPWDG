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
import scipy.linalg as sl

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

@put.print_timing
def pwprod(g, qxw, k, theta):
    dirs = np.vstack([np.cos(theta), np.sin(theta)])
    ddirs = np.vstack([-dirs[1], dirs[0]])
    
    qx, qw = qxw    
    pw = pcb.PlaneWaves(dirs.T, k)
    pwv = pw.values(qx)
    gv = g.values(qx).reshape(-1,1) # * (1+qx[:,0]*2 + qx[:,1]).reshape(-1,1)
    gvqw = gv * qw.reshape(-1,1)    
    pwnorm2 = sum(qw)
    gnorm2 = np.dot(gvqw.conj().T,  gv)    
    
    qxdd = np.dot(qx,ddirs)
    qxd = np.dot(qx, dirs)
    
    ip = np.dot(gvqw.conj().T,pwv)
    proj = np.abs(ip * ip.conj() / (gnorm2 * pwnorm2))
    ipd = np.dot(gvqw.conj().T,  pwv * 1j * k * qxdd)
    projd = 2 * np.real(ipd * ip.conj() / (gnorm2 * pwnorm2))
    ipdd = np.dot(gvqw.conj().T, pwv * (-1j * k * qxd - k**2 * qxdd*qxdd.conj()))
    projdd = 2 * np.real((ipdd * ip.conj() + ipd * ipd.conj())/ (gnorm2 * pwnorm2))
    return proj, projd, projdd

def pwproduniform(g, qxw, k, n):
    theta = np.linspace(0, 2*math.pi, n, endpoint=False)
    return (theta,)+ pwprod(g, qxw, k, theta)

@put.print_timing
def findpw(g, qxw, k, diameter, threshold = 0.2, maxtheta = 0):
    n = 2 * int(k*diameter) # this is a bit of a guess right now 
    theta = np.linspace(0, 2*math.pi, n, endpoint = False)
    thresh2 = threshold**2
    for i in range(4):
        f, fd, fdd = pwprod(g, qxw, k, theta)
        theta = theta - fd[0] / fdd[0]
        print i, np.vstack((theta, f[0])).T
        notclose = np.abs(np.fmod((np.roll(theta,-1) - theta), 2*math.pi)) > (math.pi / (10 *n)) if len(theta) > 1 else theta==theta
        notsmall = f[0] >= thresh2 
        notmin = fdd[0] < 0
        theta = np.sort(np.fmod(theta[notsmall * notclose * notmin] + 2 * math.pi, 2*math.pi))
        if len(theta) ==0: break
#        thresh2 = max(thresh2, np.max(f[0])/2)
#        print thresh2
    if maxtheta:
        f, fd, fdd = pwprod(g, qxw, k, theta)
        idx = np.argsort(f[0])
        return theta[idx][-maxtheta:]
    else:
        return theta
        

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
#    g = pcb.FourierHankel([-1,-0.5], [10], k)
    g = pcb.BasisReduce(pcb.PlaneWaves(pcb.circleDirections(20)[[5,8]], k), [3,1])
#    g = pcb.BasisReduce(pcb.BasisCombine([pcb.FourierHankel([-1,-0.5], [0], k), pcb.FourierHankel([-0.2,0.5], [0], k)]), [1,1])
#    g = pcb.FourierBessel([0.25,0.25],[20], k)

    t1 = findpw(g, qxw, k, D, maxtheta = 1)
    g1 = project(g, qxw, k, t1)
    t2 = findpw(g1, qxw, k, D, maxtheta = 1)
    g2 = project(g1, qxw, k, t2)
    print t1
    print t2

    theta, proj, projd, projdd = pwproduniform(g2, qxw, k, 500)
#    print errs[0]
    mp.plot(theta, proj[0])
#    mp.plot(theta, projd[0])
#    mp.plot(theta, projdd[0])
#    mp.plot(theta, (np.roll(projd[0],-1) - projd[0]) / (np.roll(theta,-1) - theta))
#    mp.plot(theta, D**2 * k * np.sqrt(proj[0] / 3))
    mp.show()