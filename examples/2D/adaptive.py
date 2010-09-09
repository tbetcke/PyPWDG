'''
Created on Sep 9, 2010

@author: joel
'''
from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import Mesh
from pypwdg.core.physics import impedanceSystem
from pypwdg.core.bases import circleDirections,  PlaneWaves, FourierBessel
from pypwdg.utils.quadrature import legendrequadrature, trianglequadrature
from pypwdg.utils.timing import print_timing
from pypwdg.core.evaluation import Evaluator
from pypwdg.mesh.structure import StructureMatrices
import pypwdg.core.adaptivity as pcad

import numpy
import math

mesh_dict=gmsh_reader('../../examples/2D/square.msh')
squaremesh=Mesh(mesh_dict,dim=2)


boundaryentities = []
SM = StructureMatrices(squaremesh, boundaryentities)

k = 20
Nq = 20
Np = 6
dirs = circleDirections(Np)
#elttobasis = [[PlaneWaves(dirs, k)]] * squaremesh.nelements
fb = FourierBessel(numpy.zeros(2), numpy.arange(-3,3), k)
elttobasis = [[fb]] * squaremesh.nelements

g = PlaneWaves(numpy.array([[3.0/5,4.0/5]]), k)
#g = FourierBessel(numpy.array([-2,-1]), numpy.array([5]),k)

triquad = trianglequadrature(6)

for n in range(4):
    S,G = impedanceSystem(squaremesh, SM, k, g, legendrequadrature(Nq), elttobasis)
    
    X = spsolve(S.tocsr(), G)
    
    eval_fun=lambda points: numpy.real(Evaluator(squaremesh,elttobasis,points[:,:2]).evaluate(X))
    points = numpy.mgrid[0:1:0.05,0:1:0.05].reshape(2,-1).transpose()
        
    e = Evaluator(squaremesh, elttobasis, points)
        
    xp = e.evaluate(X)        
    gp = g.values(points).flatten()        
    ep = gp - xp
    print "L2 error", math.sqrt(numpy.vdot(ep,ep)/ len(points))
    gen, ini = pcad.pwbasisgeneration(k, 3)
    elttobasis = pcad.generatebasis(squaremesh, elttobasis, X, gen, ini, triquad)
    fb = FourierBessel(numpy.zeros(2), numpy.arange(-2,2), k)
    for bs in elttobasis: bs.append(fb)
    
    