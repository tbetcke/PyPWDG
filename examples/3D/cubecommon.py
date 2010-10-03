'''
Created on Aug 10, 2010

@author: joel
'''
from pypwdg.core.bases import cubeDirections, cubeRotations, PlaneWaves
import numpy
import math
k = 3

def g(x):
    return PlaneWaves(numpy.array([[1,0]])/math.sqrt(1), k).values(x)


import pypwdg.parallel.main

from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import gmshMesh
from pypwdg.core.physics import assemble
from pypwdg.core.boundary_data import zero_impedance, dirichlet
from pypwdg.utils.quadrature import trianglequadrature
from pypwdg.utils.timing import print_timing
from pypwdg.core.evaluation import Evaluator
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid


mesh_dict=gmsh_reader('../../examples/3D/scattmesh.msh')
cubemesh=gmshMesh(mesh_dict,dim=3)
vtkgrid=VTKGrid(cubemesh)
vtkgrid.write('test.vtu')
boundaryentities = [82, 83]


Nq = 8
Np = 2
dirs = cubeRotations(cubeDirections(Np))
elttobasis = [[PlaneWaves(dirs, k)]] * cubemesh.nelements

params={'alpha':.5, 'beta':.5,'delta':.5}

l_coeffs=[-1j*k, 1]
r_coeffs=[-1j*k, 1]
      
bnddata={82:zero_impedance(k), 83:dirichlet(g) }

S, f = assemble(cubemesh, k, trianglequadrature(Nq), elttobasis, bnddata, params)

print "Solving system"

X = print_timing(spsolve)(S.tocsr(), f.tocsr())

#print X

eval_fun=lambda points: numpy.real(Evaluator(cubemesh,elttobasis,points).evaluate(X))
bounds=numpy.array([[-2,2],[-2,2],[-2,2]],dtype='d')
npoints=numpy.array([25,25,10])
vtk_structure=VTKStructuredPoints(eval_fun)
vtk_structure.create_vtk_structured_points(bounds,npoints)
vtk_structure.write_to_file('test.vti')


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
