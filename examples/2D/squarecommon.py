'''
Created on Aug 10, 2010

@author: joel
'''
import pypwdg.parallel.main

from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import Mesh
from pypwdg.core.physics import impedanceSystem
from pypwdg.core.bases import circleDirections,  PlaneWaves
from pypwdg.utils.quadrature import legendrequadrature
from pypwdg.utils.timing import print_timing
from pypwdg.core.evaluation import Evaluator
from pypwdg.mesh.structure import StructureMatrices
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid

import numpy
import math

mesh_dict=gmsh_reader('../../examples/2D/square.msh')
squaremesh=Mesh(mesh_dict,dim=2)
vtkgrid=VTKGrid(squaremesh)
vtkgrid.write('test2d.vtu')


boundaryentities = []
SM = StructureMatrices(squaremesh, boundaryentities)

k = 20
Nq = 20
Np = 12
dirs = circleDirections(Np)
elttobasis = [[PlaneWaves(dirs, k)]] * squaremesh.nelements

g = PlaneWaves(numpy.array([[3.0/5,4.0/5]]), k)

S,G = print_timing(impedanceSystem)(squaremesh, SM, k, g, legendrequadrature(Nq), elttobasis)

X = print_timing(spsolve)(S.tocsr(), G)


def eval_fun(points):
    E = Evaluator(squaremesh,elttobasis,points[:,:2])
    return numpy.real(E.evaluate(X))
bounds=numpy.array([[0,1],[0,1],[0,0]],dtype='d')
npoints=numpy.array([200,200,1])
vtk_structure=VTKStructuredPoints(eval_fun)
vtk_structure.create_vtk_structured_points(bounds,npoints)
vtk_structure.write_to_file('test2d.vti')


points = numpy.mgrid[0:1:0.01,0:1:0.01].reshape(2,-1).transpose()

e = Evaluator(squaremesh, elttobasis, points)

xp = e.evaluate(X)

gp = g.values(points).flatten()

ep = gp - xp

#print ep
print math.sqrt(numpy.vdot(ep,ep)/ len(points))
