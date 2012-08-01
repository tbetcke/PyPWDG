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
import logging

prw.log.setLevel(logging.INFO)
prw.log.addHandler(logging.StreamHandler())

class NicolasBubble(object):
    ''' Defines a function that is constant outside a circle, within which it's a bubble'''
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
    ''' The cheap way to do interpolation '''
    phases, points = map(np.concatenate, zip(*[(t*np.ones(len(x)), x) for t, (x, _) in enumerate(wfs)])) # really sorry about this - simple pleasures for simple minds
    return si.LinearNDInterpolator(points, phases*dt, fill_value = -1)
            

def bubblematerial(c = 1, N = 20, dt = 0.05, gridpoints = 100, plotoutput = True):
    ''' Calculate first arrival times for a wave travelling through NicolasBubble using a wavefront method
        that is then interpolated onto a grid
        Args:
            c: baseline wave speed
            N: number of points to put on initial wavefront
            dt: timestep (the wavefront implementation uses forward Euler)
            gridpoints: number of points to resolve on the grid
            plotoutput: whether to display output
        Returns:
            A 2D array containing the first arrival times at the points on the grid
            
    '''
    bounds = [[0,1],[0,1]] # The x-axis and y-axis bounds for the domain
    npoints = [gridpoints,gridpoints] # The number of points to (eventually) resolve in the grid 
    
    speed = NicolasBubble()
    if plotoutput: pom.output2dfn(bounds, speed, npoints, show=False) # plot the speed function
    slowness = lambda x : 1/ speed(x) 
    gradslowness = prw.gradient(slowness, 1E-6) # A very crude numerical gradient

#    These two lines would initialise a plane wave entering the domain
#    x0 = np.vstack((np.linspace(0,1,N), np.zeros(N))).T
#    p0 = np.vstack((np.zeros(N), np.ones(N))).T

#    These three lines initialise a source
    angles = np.linspace(-1E-2, math.pi+1E-2, N)
    p0 = np.vstack((np.cos(angles), np.sin(angles))).T
    x0 = np.array([0.5,0]) + dt * p0

#    Perform the wavefront tracking 
    wfs, idxs = prw.wavefront(x0, p0, slowness, gradslowness, dt, 1.25/c, 0.1)
    if plotoutput: erw.plotwavefront(wfs, idxs, bounds) # plot the wavefronts    

#    Use SciPy's interpolation (which doesn't work well when there are multiple arrival times)
#    phasefn = interpolatephase(wfs, dt)
#    pom.output2dfn(bounds, phasefn, npoints, show=False, type='contour')

#    Home-brew interpolation:
    sp = pug.StructuredPoints(np.array(bounds).T, npoints) # Define the points onto which we're going to interpolate
    initialbox = [[0.4,0],[0.6,0.1]] # The vertices of a box that contain (some of) the first wave front
    pointinfo = prw.StructuredPointInfo(sp, sp.getPoints(initialbox)[0]) # Obtain the indices of the points that are in the initial box
    h = np.max((sp.upper - sp.lower) / sp.npoints) # The (maximum) grid spacing 
    _, phases = prw.nodesToDirsAndPhases(wfs, idxs, pointinfo, lookback = int(math.ceil(h / (c * dt)))) # perform the interpolation
    
    firstphase = np.array([p[0] if len(p) > 0 else -1 for p in phases])*dt # We only care about the first phase found per point

#    pom.image(firstphase, (M+1,M+1), np.array(bounds))
    if plotoutput: pom.contour(pointinfo.points, firstphase, npoints)
    return firstphase.reshape(npoints)
    
if __name__=="__main__":
    bubblematerial()
    mp.show()