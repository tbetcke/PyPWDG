'''
Created on Feb 1, 2011

@author: joel
'''
import pypwdg.core.bases.utilities as pcbu
import pypwdg.utils.geometry as pug
import numpy as np
import matplotlib.pyplot as mp

def image(v, npoints, bounds, alpha = 1.0, cmap=None):    
    z = v.reshape(npoints)
    
    mp.figure()
    print bounds.ravel()
    c = mp.imshow(z.T, extent=bounds.ravel(), origin='lower', alpha = alpha, cmap=cmap)
    mp.colorbar(c)


def contour(p, v, npoints):    
    x = p[:,0].reshape(npoints)
    y = p[:,1].reshape(npoints)
    z = v.reshape(npoints)
    
    mp.figure()
    c = mp.contourf(x,y,z)
    mp.colorbar(c)

def showmesh(mesh):    
    mp.triplot(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements, linewidth=1.0, color='k')

def output2derror(bounds, solution, fn, npoints, plotmesh = True):
    bounds=np.array(bounds,dtype='d')
    points = pug.StructuredPoints(bounds.transpose(), npoints)
    spe = solution.getEvaluator(filter=lambda x:x)
    
    vals, counts = spe.evaluate(points)
    counts[counts==0] = 1
    vals /= counts
    fv = fn(points.toArray()).squeeze()
    
    image(np.log10(np.abs(vals - fv)), npoints, bounds)
    if plotmesh: showmesh(solution.problem.mesh)
    mp.show()
    

def output2dsoln(bounds, solution, npoints, filter = np.real, plotmesh = True, show = True, **kwargs):    
    bounds=np.array(bounds,dtype='d')
    points = pug.StructuredPoints(bounds.transpose(), npoints)
    spe = solution.getEvaluator(filter)
    
    vals, counts = spe.evaluate(points)
    counts[counts==0] = 1
    vals /= counts
#    contour(points.toArray(), vals, npoints)
    image(vals, npoints, bounds, **kwargs)
    if plotmesh: showmesh(solution.computation.problem.mesh)
    if show: mp.show()
       
def output2dfn(bounds, fn, npoints, show = True, **kwargs):
    bounds=np.array(bounds,dtype='d')
    points = pug.StructuredPoints(bounds.transpose(), npoints).toArray()
    v = np.real(fn(points))
#    contour(points, v, npoints)
    image(v, npoints, bounds, **kwargs)
    if show: mp.show()

def showdirections2(mesh, etods, **kwargs):
    ''' Todo: combine this with showdirections'''
    elementinfo = pcbu.ElementInfo(mesh, 0)
    centres = []
    directions = []
    for e in range(mesh.nelements):
        c = elementinfo.origin(e)
        for d in etods[e]:
            print d                    
            centres.append(c)
            directions.append(d)
    centres = np.array(centres)
    directions = np.array(directions)
    print centres.shape, directions.shape
    mp.quiver(centres[:,0], centres[:,1], directions[:,0], directions[:,1], **kwargs)    
    
def showdirections(mesh, etob, **kwargs):
    elementinfo = pcbu.ElementInfo(mesh, 0)
    centres = []
    directions = []
    for e in range(mesh.nelements):
        c = elementinfo.origin(e)
        bs = etob[e]
        
        for b in bs:
            if hasattr(b, "directions"):
                for d in b.directions.transpose():                    
                    centres.append(c)
                    directions.append(d)
    centres = np.array(centres)
    directions = np.array(directions)
    print centres.shape, directions.shape
    mp.quiver(centres[:,0], centres[:,1], directions[:,0], directions[:,1], **kwargs)