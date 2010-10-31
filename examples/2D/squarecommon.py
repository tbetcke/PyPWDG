'''
Created on Aug 10, 2010

@author: joel
'''
from pypwdg.core.bases import circleDirections, PlaneWaves
import numpy
import math
k = 5

# This has to be before pypwdg.parallel.main because we're serialising it to the workers.
# If it occurs later, the workers can't deserialise it.  It's a bit hacky, but I expect
# that we'll be moving this out of here soon anyway when we do the general start scripts.
def g(x):
    return PlaneWaves(numpy.array([[1,0]])/math.sqrt(1+0), k).values(x)

import pypwdg.parallel.main

from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import gmshMesh
from pypwdg.core.physics import assemble
from pypwdg.core.boundary_data import zero_impedance, dirichlet
from pypwdg.utils.quadrature import legendrequadrature
from pypwdg.utils.timing import print_timing
from pypwdg.core.evaluation import Evaluator, EvalElementError
#from pypwdg.mesh.structure import StructureMatrices
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid


mesh_dict=gmsh_reader('../../examples/2D/square.msh')
mesh=gmshMesh(mesh_dict,dim=2)

#mesh.partition(4)
#print cubemesh.nodes
    

boundaryentities = [7,8]
#SM = StructureMatrices(mesh, boundaryentities)

Nq = 20
Np = 10
dirs = circleDirections(Np)
elttobasis = [[PlaneWaves(dirs, k)]] * mesh.nelements

params={'alpha':.5, 'beta':.5,'delta':.5}
      
bnddata={7:dirichlet(g), 
         8:dirichlet(g)}

quad=legendrequadrature(Nq)
S, f, vandermondes, bndv = assemble(mesh, k, quad,elttobasis, bnddata, params)

print "Solving system"


from pymklpardiso.linsolve import solve

S=S.tocsr()
f=numpy.array(f.todense())
f=f.squeeze()
(X,error)=solve(S,f)


#S=S.tocsr()
#f=f.tocsr()
#X = print_timing(spsolve)(S, f)

print "Residual: %e: " % numpy.linalg.norm(S*X-f) 

#print X

EvalError=EvalElementError(mesh,elttobasis,quad, bnddata, vandermondes, bndv)
(ed,en,eb)=EvalError.evaluate(X)
print numpy.linalg.norm(ed),numpy.linalg.norm(en),numpy.linalg.norm(eb)

vtkgrid=VTKGrid(mesh,scalars=ed)
vtkgrid.write('square.vtu')


print "Evaluating solution"

eval_fun=lambda points: numpy.real(Evaluator(mesh,elttobasis,points[:,:2]).evaluate(X))
bounds=numpy.array([[0,1],[0,1],[0,0]],dtype='d')
npoints=numpy.array([100,100,1])
vtk_structure=VTKStructuredPoints(eval_fun)
vtk_structure.create_vtk_structured_points(bounds,npoints)
vtk_structure.write_to_file('square.vti')
