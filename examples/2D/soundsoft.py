from pypwdg import PlaneWaves, planeWaveBases
from pypwdg import setup, gmshMesh
from pypwdg import zero_dirichlet,generic_boundary_data 
from numpy import array,sqrt

k = 15
direction=array([[1.0,1.0]])/sqrt(2)
g = PlaneWaves(direction, k)

bnddata={11:zero_dirichlet(),
         10:generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[-2,2],[-2,2]],dtype='d')
npoints=array([200,200])

mesh = gmshMesh('squarescatt.msh',dim=2)
bases = planeWaveBases(mesh,k,nplanewaves=15)
comp=setup(mesh,k,20,bases,bnddata, True)

comp.writeSolution(bounds,npoints,fname='soundsoft.vti')
comp.writeMesh(fname='soundsoft.vtu',scalars=comp.combinedError())
