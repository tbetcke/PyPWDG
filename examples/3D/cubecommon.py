'''
Created on Aug 10, 2010

@author: joel
'''
import pypwdg.parallel.main

from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import Mesh
from pypwdg.core.physics import init_assembly, assemble_bnd, assemble_int_faces
from pypwdg.core.bases import cubeDirections, cubeRotations, PlaneWaves
from pypwdg.core.boundary_data import generic_boundary_data, zero_impedance, dirichlet
from pypwdg.utils.quadrature import trianglequadrature
from pypwdg.utils.timing import print_timing
from pypwdg.core.evaluation import Evaluator
from pypwdg.mesh.structure import StructureMatrices
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid
import numpy
import math


mesh_dict=gmsh_reader('../../examples/3D/scattmesh.msh')
cubemesh=Mesh(mesh_dict,dim=3)
cubemesh.partition(4)
#print cubemesh.nodes
vtkgrid=VTKGrid(cubemesh)
vtkgrid.write('test.vtu')
    

boundaryentities = [82, 83]
SM = StructureMatrices(cubemesh, boundaryentities)

k = 3

Nq = 8
Np = 2
dirs = cubeRotations(cubeDirections(Np))
elttobasis = [[PlaneWaves(dirs, k)]] * cubemesh.nelements

params={'alpha':.5, 'beta':.5,'delta':.5}

g = PlaneWaves(numpy.array([[1,0,0]])/math.sqrt(1), k)
l_coeffs=[-1j*k, 1]
r_coeffs=[-1j*k, 1]
      
bnddata={82:zero_impedance(k), 83:dirichlet(g.values) }


stiffassembly,loadassembly=init_assembly(cubemesh,trianglequadrature(Nq),elttobasis,bnddata,usecache=True)
Si=assemble_int_faces(cubemesh, SM, k, stiffassembly, params)
S_imp,f_imp=assemble_bnd(cubemesh, SM, k, bnddata, 82, stiffassembly, loadassembly, params)
S_d,f_d=assemble_bnd(cubemesh, SM, k, bnddata, 83, stiffassembly, loadassembly, params)
S=Si+S_imp+S_d
f=f_imp+f_d




print "Solving system"

X = print_timing(spsolve)(S.tocsr(), f)

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
