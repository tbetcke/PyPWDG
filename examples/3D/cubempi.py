'''
Created on Aug 10, 2010

@author: joel
'''
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from scipy.sparse.linalg.isolve import bicgstab, bicg 
from scipy.sparse.linalg import LinearOperator
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.mesh.mesh import Mesh
from pypwdg.core.physics import impedanceSystem
from pypwdg.core.bases import cubeDirections, cubeRotations, PlaneWaves
from pypwdg.utils.quadrature import trianglequadrature
from pypwdg.utils.timing import print_timing, Timer
from pypwdg.core.evaluation import Evaluator
from pypwdg.mesh.structure import StructureMatrices
from pypwdg.utils.parallel import MPIStructure, mpiloop
 
import boostmpi # for debugging only 
 
import numpy
import math

t = Timer().start()
mesh_dict=gmsh_reader('../../examples/3D/cube.msh')
t.split("Loaded mesh")
cubemesh=Mesh(mesh_dict,dim=3)
t.split("Built Mesh object")
mpi = MPIStructure(cubemesh)
t.split("Determined partitions")
boundaryentities = []
SM = StructureMatrices(cubemesh, boundaryentities, mpi.facepartition)
t.split("Built structure matrices")

k = 10
Nq = 8
Np = 2
dirs = cubeRotations(cubeDirections(Np))
elttobasis = [[PlaneWaves(dirs, k)]] * cubemesh.nelements

g = PlaneWaves(numpy.array([[1,2,3]])/math.sqrt(14), k)

    
SMPI,GMPI = impedanceSystem(cubemesh, SM, k, g, trianglequadrature(Nq), elttobasis)
t.split("built system")

#S, G = impedanceSystem(cubemesh, SM, k, g, trianglequadrature(Nq), elttobasis)
#
S = mpi.combine(SMPI)
G = mpi.combine(GMPI)
t.split("reduced system")

if boostmpi.rank == 0: 
    t.show()
#    print SMPI.tocsr()
#    print "difference"
#    print (S - SMPI).tocsr()
#    print (G - GMPI)
#    
    
    print "Solving system"
    
    X = print_timing(spsolve)(S.tocsr(), G)
#
##    L = LinearOperator(S.shape, matvec = S.matmat, dtype=numpy.complex128)
#    L = S.tocsr()
#    
#    X, nits = print_timing(bicgstab)(L, G, maxiter=160)
#
#Scsr = SMPI.tocsr()
#f = mpiloop(lambda x: Scsr * x)
#
#if f:    
#    L = LinearOperator(Scsr.shape, matvec = f, dtype=numpy.complex128)
#    X, nits = print_timing(bicgstab)(L, G, maxiter=320)
#    f(None)
#    
    points = numpy.mgrid[0:1:0.05,0:1:0.05,0:1:0.05].reshape(3,-1).transpose()
    
    e = Evaluator(cubemesh, elttobasis, points)
    
    xp = e.evaluate(X)
    
    gp = g.values(points).flatten()
    
    ep = gp - xp
    
    #print ep
    print math.sqrt(numpy.vdot(ep,ep)/ len(points))
