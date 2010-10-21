'''
Created on Aug 30, 2010

@author: tbetcke
'''
from pypwdg.core.bases import circleDirections, PlaneWaves
import numpy
import math
k = 80 

# This has to be before pypwdg.parallel.main because we're serialising it to the workers.
# If it occurs later, the workers can't deserialise it.  It's a bit hacky, but I expect
# that we'll be moving this out of here soon anyway when we do the general start scripts.
def g(x):
    return PlaneWaves(numpy.array([[1,0]])/math.sqrt(1), k).values(x)

import pypwdg.parallel.main

from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import gmshMesh
from pypwdg.core.physics import assemble
from pypwdg.core.boundary_data import zero_impedance, dirichlet
from pypwdg.utils.quadrature import legendrequadrature
from pypwdg.utils.timing import print_timing
from pypwdg.core.evaluation import Evaluator
#from pypwdg.mesh.structure import StructureMatrices
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid


mesh_dict=gmsh_reader('../../examples/2D/squarescatt.msh')
mesh=gmshMesh(mesh_dict,dim=2)

#mesh.partition(4)
#print cubemesh.nodes
vtkgrid=VTKGrid(mesh)
vtkgrid.write('soundsoft.vtu')
    

boundaryentities = [10,11]
#SM = StructureMatrices(mesh, boundaryentities)

Nq = 20
Np = 60
dirs = circleDirections(Np)
elttobasis = [[PlaneWaves(dirs, k)]] * mesh.nelements

params={'alpha':.5, 'beta':.5,'delta':.5}
      
bnddata={11:dirichlet(g), 
         10:zero_impedance(k)}

S, f = assemble(mesh, k, legendrequadrature(Nq), elttobasis, bnddata, params)

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

print "Evaluating solution"

eval_fun=lambda points: numpy.real(Evaluator(mesh,elttobasis,points[:,:2]).evaluate(X))
bounds=numpy.array([[-2,2],[-2,2],[0,0]],dtype='d')
npoints=numpy.array([200,200,1])
vtk_structure=VTKStructuredPoints(eval_fun)
vtk_structure.create_vtk_structured_points(bounds,npoints)
vtk_structure.write_to_file('soundsoft.vti')
