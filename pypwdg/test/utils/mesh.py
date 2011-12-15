'''
Sample meshes for use in unit tests

Created on Dec 19, 2010

@author: joel
'''

import pypwdg.mesh.mesh as pmm
import numpy as np

from examples import __path__ as examplepath

def examplemeshes3d():
    """ Loads in a list of 3D unstructured meshes used in examples"""    
    return [pmm.gmshMesh(examplepath[0] + "/3D/" + fname, 3) for fname in ["cube.msh", "cube_coarse.msh"]]

def examplemeshes2d():
    """ Loads in a list of 2D unstructured meshes used in examples"""
    return [pmm.gmshMesh(examplepath[0] + "/2D/" + fname, 2) for fname in ["square.msh", "squarescatt.msh"]]

def meshes3d():
    return examplemeshes3d()

def meshes2d():
    return [regularsquaremesh(n) for n in [1,3,5]] + examplemeshes2d()

def regularsquaremesh(n = 1, bdytag="BDY"):
    """ Returns a mesh object for the unit square composed of n*n squares each sub-divided into 2 triangles"""
    n1 = n+1
    points = np.linspace(0,1,n1)[np.mgrid[0:n1,0:n1]].reshape(2,-1).transpose()
    lowerleftpoints = np.arange(n1*n1).reshape(n1,n1)[0:n,0:n].reshape(-1,1)
    elements = [list(e) for e in (lowerleftpoints + np.array([[0,n1,n1+1,0,1,n1+1]])).reshape(-1,3)]
    bdy1 = np.vstack((np.arange(n), np.arange(1,n1))).transpose()
    bdyfaces = np.vstack((bdy1, bdy1 + n1*n, bdy1*n1, bdy1*n1 + n))
    boundary = [(bdytag, tuple(bf)) for bf in bdyfaces]
    geomEntity=[1]*len(elements)
    return pmm.Mesh(points, elements, geomEntity, boundary, 2)

def regularrectmesh(xlims, ylims, nx, ny):
    ''' Returns a mesh object for the rectangle xlims x ylims consisting of 
        nx x ny rectangles, each subdivided into 2 triangles.
        The edges of the grid are labelled 1,2,3,4 anti-clockwise from the left-hand edge (assuming the origin is in the lower-left)
    '''
    nx1 = nx + 1
    ny1 = ny + 1
    mg1 = np.mgrid[0:nx1, 0:ny1]
    # construct the vertices
    xp = np.linspace(*xlims, num=nx1)[mg1[0]].ravel() # x coordinates of points
    yp = np.linspace(*ylims, num=ny1)[mg1[1]].ravel() # y coordinates of points
    nodes = np.vstack((xp,yp)).T
    # construct the elements
    idx = ny1 * mg1[0] + mg1[1] # the indices of the points (arranged on the grid)
    c = [idx[np.ix_(np.arange(nx)+d[0],np.arange(ny)+d[1])]for d in [[0,0],[1,0],[0,1],[1,1]]] # indices of corners of squares 
    elts = [sorted(list(e)) for e in np.dstack([c[0],c[1],c[2],c[1],c[3],c[2]]).reshape(-1,3)] # build the list of triangular elements
    # construct the boundaries.   
    cc = np.array([[c[0],c[2]],[c[1],c[3]]]) # hold tight, it's a 4-dimensional array
    bdys = [(bdyid+1, face) for bdyid, side in enumerate([cc[0,:,0,:].T,cc[:,0,:,0].T,cc[1,:,-1,:].T,cc[:,1,:,-1].T]) for face in side]
    geomEntity = [5]*len(elts)
    return pmm.Mesh(nodes, elts, geomEntity, bdys, 2)
    
    
    
    
        