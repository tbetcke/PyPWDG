'''
Created on Aug 10, 2010

@author: joel
'''
from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from PyPWDG.mesh.gmsh_reader import gmsh_reader
from PyPWDG.mesh.mesh import Mesh
from PyPWDG.core.physics import impedanceSystem
from PyPWDG.core.bases import circleDirections,  PlaneWaves
from PyPWDG.utils.quadrature import legendrequadrature
from PyPWDG.utils.timing import print_timing
from PyPWDG.core.evaluation import Evaluator
import numpy

mesh_dict=gmsh_reader('../../examples/2D/square.msh')
squaremesh=Mesh(mesh_dict,dim=2)
k = 20
Nq = 10
Np = 8
dirs = circleDirections(Np)
elttobasis = [[PlaneWaves(dirs, k)]] * squaremesh.nelements

g = PlaneWaves(numpy.array([[1,0]]), k)

S,G = impedanceSystem(squaremesh, k, g, legendrequadrature(Nq), elttobasis)

X = print_timing(spsolve)(S.tocsr(), G.tocsr().todense())

points = numpy.mgrid[0:1:0.1,0:1:0.1].reshape(2,-1).transpose()

e = Evaluator(squaremesh, elttobasis, points)

xp = e.evaluate(X)

gp = g.values(points).flatten()

print gp - xp
