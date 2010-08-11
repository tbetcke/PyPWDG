'''
Created on Aug 10, 2010

@author: joel
'''
from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from PyPWDG.mesh.gmsh_reader import gmsh_reader
from PyPWDG.mesh.mesh import Mesh
from PyPWDG.core.physics import impedanceSystem
from PyPWDG.core.bases import cubeDirections, cubeRotations, PlaneWaves
from PyPWDG.utils.quadrature import trianglequadrature
from PyPWDG.utils.timing import print_timing
from PyPWDG.core.evaluation import Evaluator
import numpy
import math


mesh_dict=gmsh_reader('../../examples/3D/cube.msh')
cubemesh=Mesh(mesh_dict,dim=3)
k = 10
Nq = 8
Np = 2
dirs = cubeRotations(cubeDirections(Np))
elttobasis = [[PlaneWaves(dirs, k)]] * cubemesh.nelements

g = PlaneWaves(numpy.array([[1,2,3]])/math.sqrt(14), k)

S,G = impedanceSystem(cubemesh, k, g, trianglequadrature(Nq), elttobasis)

print "Solving system"

X = print_timing(spsolve)(S.tocsr(), G.tocsr().todense())

points = numpy.mgrid[0:1:0.2,0:1:0.2,0:1:0.02].reshape(3,-1).transpose()

e = Evaluator(cubemesh, elttobasis, points)

xp = e.evaluate(X)

gp = g.values(points).flatten()

ep = gp - xp

#print ep
print math.sqrt(numpy.vdot(ep,ep)/ len(points))
