'''
Created on Jul 26, 2012

@author: joel
'''
import matplotlib.pyplot as mp
import pypwdg.raytrace.wavefront as prw
import pypwdg.output.mploutput as pom
import pypwdg.core.bases as pcb
import pypwdg.raytrace.basisrules as prb
import pypwdg.test.utils.mesh as ptum
import experiments.raytracing.wavefrontexamples as erw
import numpy as np
import math

class GaussianBubble:
    def __init__(self, c = 1, O = [0.5,0.3]):
        self.c = c
        self.O = O
    
    def __call__(self,x):
        r2 = np.sum((x - self.O)**2, axis=1)                
        return 1.0 / ((1- np.exp(-32*r2)/2) * self.c)

class NicolasBubble:
    def __init__(self, c=1, origin = [0.5,0.5], radius = 0.25):
        self.c = c
        self.origin = origin
        self.radius2 = radius * radius
        
    def __call__(self, x):
        r2 = np.sum((x - self.origin)**2, axis=1)
        scaledr2 = r2 / self.radius2
    
    
def hump(c = 1, yc = 0.3, yr = 0.1, alpha = 0.3):
    def slowness(x):
        yparabola = (x[:,1] - yc)**2 / (yr**2) - 1
        return  (1 + (yparabola < 0) * yparabola * (-alpha))*c
    return slowness, prw.gradient(slowness, 1E-6)

def bubblematerial(c = 1, N = 20):
#    slowness, gradslowness = bubble(c)
#    slowness, gradslowness = hump(c)
    slowness = GaussianBubble()
    gradslowness = prw.gradient(slowness, 1E-6)
    x0 = np.vstack((np.linspace(0,1,N), np.zeros(N))).T
    p0 = np.vstack((np.zeros(N), np.ones(N))).T
    wfs, idxs = prw.wavefront(x0, p0, slowness, gradslowness, 0.1, 1.2/c, 0.1)
    mp.subplot(1,2,1)
    erw.plotwavefront(wfs, idxs)
    meshinfo = ptum.regularsquaremeshinfo(12, "BDY")
    etob = prb.getetob(wfs, idxs, meshinfo, "BDY")
    mp.subplot(1,2,2)
    pom.showmesh(meshinfo)
#    pom.showdirections(meshinfo, etob,scale=20)
    
if __name__=="__main__":
    bubblematerial()
    mp.show()