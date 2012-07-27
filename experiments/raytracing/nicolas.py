'''
Created on Jul 26, 2012

@author: joel
'''
import matplotlib.pyplot as mp
import pypwdg.raytrace.wavefront as prw
import pypwdg.output.mploutput as pom
import pypwdg.test.utils.mesh as ptum
import wavefrontexamples as erw
import numpy as np
import scipy.interpolate as si
import math

class NicolasBubble(object):
    def __init__(self, c=1, origin = [0.5,0.5], radius = 0.25):
        self.c = c
        self.origin = origin
        self.radius2 = radius * radius
        
    def __call__(self, x):
        r2 = np.sum((x - self.origin)**2, axis=1)
        scaledr2 = r2 / self.radius2
        speed = np.ones(len(r2))
        # Taking 0.95 rather than 1.0 in the following line avoids an overflow.  It introduces an error of about e^(-40).
        inbub = scaledr2 < 0.95
        speed[inbub] += 2.0 * np.exp(1/(np.sqrt(scaledr2[inbub])-1))
        return speed * self.c

def interpolatephase(wfs, dt):
    phases, points = map(np.concatenate, zip(*[(t*np.ones(len(x)), x) for t, (x, _) in enumerate(wfs)])) # really sorry about this - simple pleasures for simple minds
    return si.LinearNDInterpolator(points, phases*dt, fill_value = -1)
            

def bubblematerial(c = 1, N = 20, dt = 0.05):
    
    bounds = [[0,1],[0,1]]
    npoints = [50,50]
    
    speed = NicolasBubble()
    pom.output2dfn(bounds, speed, npoints, show=False)
    slowness = lambda x : 1/ speed(x)
    gradslowness = prw.gradient(slowness, 1E-6)
#    x0 = np.vstack((np.linspace(0,1,N), np.zeros(N))).T
#    p0 = np.vstack((np.zeros(N), np.ones(N))).T
    angles = np.linspace(0, math.pi, N)
    p0 = np.vstack((np.cos(angles), np.sin(angles))).T
    x0 = np.array([0.5,0]) + dt * p0
    
    wfs, idxs = prw.wavefront(x0, p0, slowness, gradslowness, dt, 1.01/c, 0.1)
    erw.plotwavefront(wfs, idxs, bounds)    
    phasefn = interpolatephase(wfs, dt)
    pom.output2dfn(bounds, phasefn, npoints, show=False)
    
#    meshinfo = ptum.regularsquaremeshinfo(12, "BDY")
#    dirs, phases = prw.nodesToDirsAndPhases(wfs, idxs, meshinfo, ["BDY"])
#    
#    print dirs
#    print phases
#    etob = prb.getetob(wfs, idxs, meshinfo, "BDY")
#    mp.subplot(1,2,2)
#    pom.showmesh(meshinfo)
#    pom.showdirections(meshinfo, etob,scale=20)
    
if __name__=="__main__":
    bubblematerial()
    mp.show()