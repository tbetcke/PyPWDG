'''
Created on Aug 30, 2010

@author: tbetcke
'''
from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import Mesh
from pypwdg.core.physics import init_assembly, assemble_bnd, assemble_int_faces
from pypwdg.core.bases import circleDirections, PlaneWaves
from pypwdg.core.boundary_data import zero_impedance, dirichlet, generic_boundary_data
from pypwdg.utils.quadrature import legendrequadrature
from pypwdg.utils.timing import print_timing
from pypwdg.core.evaluation import Evaluator
from pypwdg.mesh.structure import StructureMatrices
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid
import numpy
import math


mesh_dict=gmsh_reader('../../examples/2D/squarescatt.msh')
mesh=Mesh(mesh_dict,dim=2)
mesh.partition(4)
#print cubemesh.nodes
vtkgrid=VTKGrid(mesh)
vtkgrid.write('soundsoft.vtu')
    

boundaryentities = [10,11]
SM = StructureMatrices(mesh, boundaryentities)

k = 40
Nq = 20
Np = 20
dirs = circleDirections(Np)
elttobasis = [[PlaneWaves(dirs, k)]] * mesh.nelements

params={'alpha':.5, 'beta':.5,'delta':.5}

g = PlaneWaves(numpy.array([[1,0]])/math.sqrt(1), k)
l_coeffs=[-1j*k, 1]
r_coeffs=[-1j*k, 1]
      
bnddata={11:dirichlet(g.values), 
         10:dirichlet(g.values)}

stiffassembly,loadassembly=init_assembly(mesh,legendrequadrature(Nq),elttobasis,bnddata,usecache=True)

S=assemble_int_faces(mesh, SM, k, stiffassembly, params)
f=0

for i in bnddata:
    print i
    (Sb,fb)=assemble_bnd(mesh, SM, k, bnddata, i, stiffassembly, loadassembly, params)
    S=S+Sb
    f=f+fb

print "Solving system"

X = print_timing(spsolve)(S.tocsr(), f)

#print X

print "Evaluating solution"

eval_fun=lambda points: numpy.real(Evaluator(mesh,elttobasis,points[:,:2]).evaluate(X))
bounds=numpy.array([[-1,1],[-1,1],[0,0]],dtype='d')
npoints=numpy.array([200,200,1])
vtk_structure=VTKStructuredPoints(eval_fun)
vtk_structure.create_vtk_structured_points(bounds,npoints)
vtk_structure.write_to_file('soundsoft.vti')