'''
Created on Feb 1, 2011

@author: joel
'''
import pypwdg.core.bases.utilities as pcbu
import pypwdg.utils.geometry as pug
import numpy as np
import matplotlib.pyplot as mp

def image(v, npoints, bounds):    
    z = v.reshape(npoints)
    
    mp.figure()
    c = mp.imshow(z.T, extent=bounds.ravel(), origin='lower', alpha = 0.7)
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
    mp.quiver(centres[:,0], centres[:,1], directions[:,0], directions[:,1], **kwargs)