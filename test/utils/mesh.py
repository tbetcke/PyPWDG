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