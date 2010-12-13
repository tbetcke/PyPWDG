from pypwdg import setup,gmshMesh, PlaneWaves, planeWaveBases
from pypwdg import zero_dirichlet, generic_boundary_data
from numpy import array

k = 3
direction=array([[1.0,0]])
g = PlaneWaves(direction, k)

bnddata={21:zero_dirichlet(),
         18:generic_boundary_data([-1j*k,1],[-1j*k,1],g),
         20:generic_boundary_data([-1j*k,1],[-1j*k,1],g),
         19:zero_dirichlet()}

bounds=array([[0,5],[0,1]],dtype='d')
npoints=array([50,10])

mesh = gmshMesh('2dhole.msh',dim=2)
bases = planeWaveBases(mesh,k,nplanewaves=10)

comp=setup(mesh,k,20,bases,bnddata=bnddata)
comp.solve()
comp.writeSolution(bounds,npoints,fname='2dhole.vti')
comp.writeMesh(fname='2dhole.vtu',scalars=comp.combinedError())