'''
Created on Aug 10, 2010

@author: joel
'''
from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from PyPWDG.mesh.gmsh_reader import gmsh_reader
from PyPWDG.mesh.mesh import Mesh
from PyPWDG.core.assembly import impedanceSystem
from PyPWDG.core.bases import cubeDirections, cubeRotations, PlaneWaves
from PyPWDG.utils.quadrature import trianglequadrature
from PyPWDG.utils.timing import print_timing
import numpy

mesh_dict=gmsh_reader('../../examples/3D/cube_coarse.msh')
cubemesh=Mesh(mesh_dict,dim=3)
k = 10
Nq = 15
Np = 3
dirs = cubeRotations(cubeDirections(Np))
g = PlaneWaves(numpy.array([[1,0,0]]), k)

S,G = impedanceSystem(cubemesh, k, g, trianglequadrature(Nq), dirs)

print "Solving system"

X = print_timing(spsolve)(S.tocsr(), G.tocsr().todense())
