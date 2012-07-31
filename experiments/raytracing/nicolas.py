'''
Created on Jul 26, 2012

@author: joel
'''
import matplotlib.pyplot as mp
import pypwdg.raytrace.wavefront as prw
import pypwdg.output.mploutput as pom
import pypwdg.test.utils.mesh as ptum
import pypwdg.utils.geometry as pug
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
            

def bubblematerial(c = 1, N = 20, dt = 0.01):
    
    bounds = [[0,1],[0,1]]
    npoints = [50,50]
    
    speed = NicolasBubble()
    pom.output2dfn(bounds, speed, npoints, show=False)
    slowness = lambda x : 1/ speed(x)
    gradslowness = prw.gradient(slowness, 1E-6)
#    x0 = np.vstack((np.linspace(0,1,N), np.zeros(N))).T
#    p0 = np.vstack((np.zeros(N), np.ones(N))).T
    angles = np.linspace(-1E-2, math.pi+1E-2, N)
    p0 = np.vstack((np.cos(angles), np.sin(angles))).T
    x0 = np.array([0.5,0]) + dt * p0
    
    wfs, idxs = prw.wavefront(x0, p0, slowness, gradslowness, dt, 1.25/c, 0.1)
    erw.plotwavefront(wfs, idxs, bounds)    
    phasefn = interpolatephase(wfs, dt)
    pom.output2dfn(bounds, phasefn, npoints, show=False, type='contour')
    
#    M = 100
#    meshinfo = ptum.regularsquaremeshinfo(M, "BDY")
#    _, phases = prw.nodesToDirsAndPhases(wfs, idxs, prw.MeshPointInfo(meshinfo, ["BDY"]))
    
    sp = pug.StructuredPoints(np.array(bounds).T, npoints)
    pointinfo = prw.StructuredPointInfo(sp, sp.getPoints([[0.4,-0.1],[0.6,0.1]])[0])
    print "boundary", pointinfo.points[pointinfo.boundary()]
    _, phases = prw.nodesToDirsAndPhases(wfs, idxs, pointinfo)
    
    
    firstphase = np.array([p[0] if len(p) > 0 else -1 for p in phases])*dt
    print len(firstphase)
#    pom.image(firstphase, (M+1,M+1), np.array(bounds))
    pom.contour(pointinfo.points, firstphase, npoints)

    
if __name__=="__main__":
    bubblematerial()
    mp.show()