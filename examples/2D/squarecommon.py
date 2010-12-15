from pypwdg import PlaneWaves
from pypwdg import setup, gmshMesh, planeWaveBases
from pypwdg import generic_boundary_data, dirichlet
from numpy import array,sqrt

k = 60
direction=array([[1.0,1.0]])/sqrt(2)
g = PlaneWaves(direction, k)
impbd = generic_boundary_data([-1j*k,1],[-1j*k,1],g)

#bnddata={7:impbd, 
#         8:impbd}
bnddata={7:dirichlet(g), 
         8:dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([100,100])

mesh = gmshMesh('square.msh',dim=2)
bases = planeWaveBases(mesh,k,nplanewaves=15)
comp=setup(mesh,k,20, bases, bnddata)
comp.writeSolution(bounds,npoints,fname='square.vti')
comp.writeMesh(fname='square.vtu',scalars=comp.combinedError())

