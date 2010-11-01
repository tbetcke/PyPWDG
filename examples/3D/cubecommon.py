'''
Created on Aug 10, 2010

@author: joel
'''
from pypwdg.core.bases import cubeDirections, cubeRotations, PlaneWaves
import numpy
import math
k = 10

def g(x):
    return PlaneWaves(numpy.array([[1,0,0]])/math.sqrt(1), k).values(x)


import pypwdg.parallel.main

from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import gmshMesh
from pypwdg.core.physics import assemble
from pypwdg.core.boundary_data import zero_impedance, dirichlet
from pypwdg.utils.quadrature import trianglequadrature
from pypwdg.utils.timing import print_timing
from pypwdg.core.evaluation import Evaluator, EvalElementError
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid
from pypwdg.mesh.meshutils import MeshQuadratures
from pypwdg.core.vandermonde import LocalVandermondes



mesh_dict=gmsh_reader('../../examples/3D/scattmesh.msh')
mesh=gmshMesh(mesh_dict,dim=3)
boundaryentities = [82, 83]


Nq = 16
Np = 3
dirs = cubeRotations(cubeDirections(Np))
quad=trianglequadrature(Nq)


elttobasis = [[PlaneWaves(dirs, k)]] * mesh.nelements

params={'alpha':.5, 'beta':.5,'delta':.5}

l_coeffs=[-1j*k, 1]
r_coeffs=[-1j*k, 1]
      
bnddata={82:zero_impedance(k), 83:dirichlet(g) }

mqs = MeshQuadratures(mesh, quad)
lv = LocalVandermondes(mesh, elttobasis, mqs, usecache=False)
bndvs=[]
for data in bnddata.values():
    bndv = LocalVandermondes(mesh, [[data]] * mesh.nelements, mqs)        
    bndvs.append(bndv)


S, f = assemble(mesh, k, lv, bndvs, mqs, elttobasis, bnddata, params)


print "Solving system"


from pymklpardiso.linsolve import solve

S=S.tocsr()
f=numpy.array(f.todense())
f=f.squeeze()
(X,error)=solve(S,f)


#X = print_timing(spsolve)(S.tocsr(), f.tocsr())

#print X


print "Residual: %e: " % numpy.linalg.norm(S*X-f) 



EvalError=EvalElementError(mesh,elttobasis,quad, bnddata, lv, bndvs)
(ed,en,eb)=EvalError.evaluate(X)
print numpy.linalg.norm(ed),numpy.linalg.norm(en),numpy.linalg.norm(eb)

vtkgrid=VTKGrid(mesh,scalars=ed)
vtkgrid.write('test.vtu')


eval_fun=lambda points: numpy.real(Evaluator(mesh,elttobasis,points).evaluate(X))
bounds=numpy.array([[-2,2],[-2,2],[-2,2]],dtype='d')
npoints=numpy.array([50,50,50])
vtk_structure=VTKStructuredPoints(eval_fun)
vtk_structure.create_vtk_structured_points(bounds,npoints)
vtk_structure.write_to_file('test.vti')

