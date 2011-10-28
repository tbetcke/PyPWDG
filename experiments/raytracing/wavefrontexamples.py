'''
Created on Oct 28, 2011

@author: joel
'''
import matplotlib.pyplot as mp
import pypwdg.raytrace.wavefront as prw
import numpy as np

def plotwavefront(wavefronts, forwardidxs = None):
    for (x,p) in wavefronts:
        mp.plot(x[:,0], x[:,1])

def trivialwavefront(c, N = 50):
    slowness = lambda x: np.ones(len(x)) /c
    gradslowness = lambda x: np.zeros_like(x)
    x0 = np.vstack((np.linspace(0,1,N), np.zeros(N))).T
    p0 = np.vstack((np.zeros(N), np.ones(N))).T
    print p0.shape
    wfs, idxs = prw.wavefront(x0, p0, slowness, gradslowness, 0.05, 20, 1)
    plotwavefront(wfs, idxs)

