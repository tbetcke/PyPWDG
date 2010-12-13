from pypwdg import PlaneWaves
from pypwdg import setup,gmshMesh, planeWaveBasis
from pypwdg import generic_boundary_data, dirichlet
from numpy import array,sqrt

k = 15
direction=array([[1.0,1.0]])/sqrt(2)
g = PlaneWaves(direction, k)
#impbd = generic_boundary_data([-1j*k,1],[-1j*k,1],g)

#bnddata={7:impd), 
#         8:impd)}
bnddata={7:dirichlet(g), 
         8:dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([100,100])

mesh = gmshMesh('square.msh',dim=2)
basis = planeWaveBasis(mesh,k,nplanewaves=15)
comp=setup(mesh,k,20, basis, bnddata)
comp.writeSolution(bounds,npoints,fname='square.vti')
comp.writeMesh(fname='square.vtu',scalars=comp.combinedError())

