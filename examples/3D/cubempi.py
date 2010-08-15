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
from pypwdg.utils.timing import print_timing
from pypwdg.core.evaluation import Evaluator
from pypwdg.mesh.structure import StructureMatrices
from pypwdg.utils.parallel import MPIStructure
 
import numpy
import math


mesh_dict=gmsh_reader('../../examples/3D/cube.msh')
cubemesh=Mesh(mesh_dict,dim=3)
mpi = MPIStructure(cubemesh)

boundaryentities = []
SM = StructureMatrices(cubemesh, boundaryentities, mpi.facepartition)


k = 10
Nq = 8
Np = 2
dirs = cubeRotations(cubeDirections(Np))
elttobasis = [[PlaneWaves(dirs, k)]] * cubemesh.nelements

g = PlaneWaves(numpy.array([[1,2,3]])/math.sqrt(14), k)

    
SMPI,GMPI = impedanceSystem(cubemesh, SM, k, g, trianglequadrature(Nq), elttobasis)
#S, G = impedanceSystem(cubemesh, SM, k, g, trianglequadrature(Nq), elttobasis)

S = mpi.combine(SMPI)
G = mpi.combine(GMPI)

if (not S is None):
#    print SMPI.tocsr()
#    print "difference"
#    print (S - SMPI).tocsr()
#    print (G - GMPI)
#    
    
    print "Solving system"
    
    X = print_timing(spsolve)(S.tocsr(), G)
    
    points = numpy.mgrid[0:1:0.2,0:1:0.2,0:1:0.02].reshape(3,-1).transpose()
    
    e = Evaluator(cubemesh, elttobasis, points)
    
    xp = e.evaluate(X)
    
    gp = g.values(points).flatten()
    
    ep = gp - xp
    
    #print ep
    print math.sqrt(numpy.vdot(ep,ep)/ len(points))
