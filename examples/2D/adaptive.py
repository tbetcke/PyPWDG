'''
Created on Sep 9, 2010

@author: joel
'''
from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import Mesh
from pypwdg.core.physics import impedanceSystem
from pypwdg.core.bases import circleDirections,  PlaneWaves, FourierBessel, FourierHankel, BasisReduce, BasisCombine
from pypwdg.utils.quadrature import legendrequadrature, trianglequadrature
from pypwdg.utils.timing import print_timing
from pypwdg.core.evaluation import Evaluator
from pypwdg.mesh.structure import StructureMatrices
import pypwdg.core.adaptivity as pcad
import pypwdg.mesh.meshutils as pmmu

from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid


import numpy
import math

mesh_dict=gmsh_reader('../../examples/2D/square.msh')
squaremesh=Mesh(mesh_dict,dim=2)


boundaryentities = []
SM = StructureMatrices(squaremesh, boundaryentities)

eltcentres = list(pmmu.elementcentres(squaremesh))
print eltcentres

k = 100
Nq = 20
Np = 3
dirs = circleDirections(Np)
#elttobasis = [[PlaneWaves(dirs, k)]] * squaremesh.nelements
#fb = FourierBessel(numpy.zeros(2), numpy.arange(0,1), k)
elttobasis = [[FourierBessel(c, numpy.arange(-1,2), k), PlaneWaves(dirs, k)] for c in eltcentres]

#g = PlaneWaves(numpy.array([[3.0/5,4.0/5]]), k)
#g = FourierBessel(numpy.array([-2,-1]), numpy.array([3]),k)
#g = FourierHankel(numpy.array([-0.5,-0.2]), numpy.array([0]),k)


        
#g = BasisComb(PlaneWaves(numpy.array([[1.0,0],[-0.6,0.8]]), k), numpy.array([2,1]))
#bases = [FourierHankel(c, numpy.array([0]), k) for c in numpy.array([[-0.2,0.5],[1.1,0.2]])]
#g = BasisCombine(bases, numpy.array([2,1]))

g = FourierHankel(numpy.array([-2.0, -1.0]), numpy.array([0]), k)

triquad = trianglequadrature(10)
bounds=numpy.array([[0,1],[0,1],[0,0]],dtype='d')
npoints=numpy.array([200,200,1])

filename = "adaptive"

for n in range(20):
    print n
    S,G = impedanceSystem(squaremesh, SM, k, g, legendrequadrature(Nq), elttobasis)
    
    X = spsolve(S.tocsr(), G)
    
    eval_fun=lambda points: numpy.real(Evaluator(squaremesh,elttobasis,points[:,:2]).evaluate(X))
    points = numpy.mgrid[0:1:0.05,0:1:0.05].reshape(2,-1).transpose()
        
    e = Evaluator(squaremesh, elttobasis, points)
    
    eval_fun=lambda points: numpy.real(Evaluator(squaremesh,elttobasis,points[:,:2]).evaluate(X))
    vtk_structure=VTKStructuredPoints(eval_fun)
    vtk_structure.create_vtk_structured_points(bounds,npoints)
    vtk_structure.write_to_file('%s%s.vti'%(filename,n))
    
    vtk_err = VTKStructuredPoints(lambda points: numpy.real(Evaluator(squaremesh,elttobasis,points[:,:2]).evaluate(X) - g.values(points[:,:2]).flatten()))
    vtk_err.create_vtk_structured_points(bounds, npoints)
    vtk_err.write_to_file('%serr%s.vti'%(filename,n))

    
    xp = e.evaluate(X)        
    gp = g.values(points).flatten()        
    ep = (gp - xp)
    print "L2 error", math.sqrt(numpy.vdot(ep,ep)/ len(points))
    print "Relative L2 error", math.sqrt(numpy.vdot(ep,ep)/ numpy.vdot(gp,gp))
    gen, ini = pcad.pwbasisgeneration(k, 5)
    elttobasis = pcad.generatebasis(squaremesh, elttobasis, X, gen, ini, triquad)
    for bs, c in zip(elttobasis, eltcentres): bs.append(FourierBessel(c, numpy.arange(-1,2), k))
 #   for bs, c in zip(elttobasis, eltcentres): bs.append(PlaneWaves(dirs, k))
    
    
    
    