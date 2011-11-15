'''
Created on Oct 28, 2011

@author: joel
'''
import matplotlib.pyplot as mp
import pypwdg.raytrace.wavefront as prw
import pypwdg.output.mploutput as pom
import pypwdg.core.bases as pcb
import pypwdg.raytrace.basisrules as prb
import pypwdg.test.utils.mesh as ptum
import numpy as np

import pypwdg.setup.problem as psp

def plotwavefront(wavefronts, forwardidxs = None):
    for (x,p) in wavefronts:
        mp.plot(x[:,0], x[:,1])
    if forwardidxs:
        x01s = []
        for ((x0,p0), (x1,p1), fidx) in zip(wavefronts[:-1], wavefronts[1:], forwardidxs[1:]):
            print x1.shape, x0.shape
            print fidx
            xf = x1 if fidx is None else x1[fidx]
            x01s.append(np.dstack((x0,xf)))
        xy = np.vstack(x01s)
        mp.plot(xy[:,0,:].T, xy[:,1,:].T, 'k:')

def trivialwavefront(c, N = 50):
    slowness = lambda x: np.ones(len(x)) /c
    gradslowness = lambda x: np.zeros_like(x)
    x0 = np.vstack((np.linspace(0,1,N), np.zeros(N))).T
    p0 = np.vstack((np.zeros(N), np.ones(N))).T
    wfs, idxs = prw.wavefront(x0, p0, slowness, gradslowness, 0.05, 1.0/c, 1)
    plotwavefront(wfs, idxs)

def linearmaterial():
    N = 50
    speed = lambda x: 1 + x[:,0]
    slowness = lambda x: 1 / speed(x)
    gradslowness = prw.gradient(slowness, 1E-6)
    x0 = np.vstack((np.linspace(0,1,N), np.zeros(N))).T
    p0 = np.vstack((np.zeros(N), np.ones(N))).T
    wfs, idxs = prw.wavefront(x0, p0, slowness, gradslowness, 0.05, 1.5, 1)
    plotwavefront(wfs, idxs)

class bubble:
    def __init__(self, c = 1, R = 0.2, alpha = 0.3, O = [0.5,0.3]):
        self.R2 = R**2
        self.c = c
        self.alpha = alpha
        self.O = O
    
    def __call__(self,x):
            r2 = np.sum((x - self.O)**2, axis=1)
            return (1 + (r2 <= self.R2) * (self.R2 - r2)*self.alpha / self.R2) / self.c
    
def hump(c = 1, yc = 0.3, yr = 0.1, alpha = 0.3):
    def slowness(x):
        yparabola = (x[:,1] - yc)**2 / (yr**2) - 1
        return  (1 + (yparabola < 0) * yparabola * (-alpha))*c
    return slowness, prw.gradient(slowness, 1E-6)
    
def bubblematerial(c = 1, N = 20):
    slowness, gradslowness = bubble(c)
    x0 = np.vstack((np.linspace(0,1,N), np.zeros(N))).T
    p0 = np.vstack((np.zeros(N), np.ones(N))).T
    wfs, idxs = prw.wavefront(x0, p0, slowness, gradslowness, 0.1, 1.2/c, 0.1)
    mp.subplot(1,2,1)
    plotwavefront(wfs, idxs)
    mesh = ptum.regularsquaremesh(12, "BDY")
    etob = prb.getetob(wfs, idxs, mesh, "BDY")
    mp.subplot(1,2,2)
    pom.showmesh(mesh)
    pom.showdirections(mesh, etob,scale=20)
    
