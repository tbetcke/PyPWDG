'''
Created on Feb 1, 2011

@author: joel
'''
import pypwdg.core.evaluation as pce
import pypwdg.utils.geometry as pug
import numpy as np
import matplotlib.pyplot as mp

def image(v, npoints, bounds):    
    z = v.reshape(npoints)
    
    mp.figure()
    c = mp.imshow(z, extent=bounds.ravel(), origin='lower')
    mp.colorbar(c)


def contour(p, v, npoints):    
    x = p[:,0].reshape(npoints)
    y = p[:,1].reshape(npoints)
    z = v.reshape(npoints)
    
    mp.figure()
    c = mp.contourf(x,y,z)
    mp.colorbar(c)

def showmesh(mesh):    
    mp.triplot(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements, linewidth=0.5)

def output2dsoln(bounds, solution, npoints, filter = np.real):    
    bounds=np.array(bounds,dtype='d')
    points = pug.StructuredPoints(bounds.transpose(), npoints)
    spe = solution.getEvaluator(filter)
    
    vals, counts = spe.evaluate(points)
    counts[counts==0] = 1
    vals /= counts
#    contour(points.toArray(), vals, npoints)
    image(vals, npoints, bounds)
    showmesh(solution.problem.mesh)
    mp.show()
       
def output2dfn(bounds, fn, npoints):
    bounds=np.array(bounds,dtype='d')
    points = pug.StructuredPoints(bounds.transpose(), npoints).toArray()
    v = np.real(fn(points))
#    contour(points, v, npoints)
    image(v, npoints, bounds)
    mp.show()