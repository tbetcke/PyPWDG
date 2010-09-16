'''
Created on Aug 10, 2010

@author: joel
'''
from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import Mesh
from pypwdg.core.physics import impedanceSystem
from pypwdg.core.bases import cubeDirections, cubeRotations, PlaneWaves
from pypwdg.utils.quadrature import trianglequadrature
from pypwdg.utils.timing import print_timing, Timer
from pypwdg.core.evaluation import Evaluator
from pypwdg.mesh.structure import StructureMatrices
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid
import numpy
import math

t = Timer().start()
mesh_dict=gmsh_reader('../../examples/3D/cube.msh')
cubemesh=Mesh(mesh_dict,dim=3)
t.split("built mesh")
#cubemesh.partition(4)
#print cubemesh.nodes
vtkgrid=VTKGrid(cubemesh)
vtkgrid.write('test.vtu')
t.split("written out mesh")

boundaryentities = []
SM = StructureMatrices(cubemesh, boundaryentities)

k = 35
Nq = 8
Np = 2
dirs = cubeRotations(cubeDirections(Np))
elttobasis = [[PlaneWaves(dirs, k)]] * cubemesh.nelements

g = PlaneWaves(numpy.array([[1,2,3]])/math.sqrt(14), k)

S,G = impedanceSystem(cubemesh, SM, k, g, trianglequadrature(Nq), elttobasis)
t.split("assembled system")
print "Solving system"

X = print_timing(spsolve)(S.tocsr(), G)
t.split("solved system")
#print X

eval_fun=lambda points: numpy.real(Evaluator(cubemesh,elttobasis,points).evaluate(X))
bounds=numpy.array([[0,1],[0,1],[0,1]],dtype='d')
npoints=numpy.array([80,80,80])
vtk_structure=VTKStructuredPoints(eval_fun)
vtk_structure.create_vtk_structured_points(bounds,npoints)
vtk_structure.write_to_file('test.vti')
t.split("output")
t.show()

#points = numpy.mgrid[0:1:0.2,0:1:0.2,0:1:0.02].reshape(3,-1).transpose()
#
#e = Evaluator(cubemesh, elttobasis, points)
#
#xp = e.evaluate(X)
#
#gp = g.values(points).flatten()
#
#ep = gp - xp
#
##print ep
#print math.sqrt(numpy.vdot(ep,ep)/ len(points))
