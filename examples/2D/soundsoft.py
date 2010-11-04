from pypwdg import PlaneWaves
from pypwdg import setup,runParallel,gmshMesh
from pypwdg import zero_dirichlet,generic_boundary_data 
from numpy import array,sqrt

k = 15
direction=array([[1.0,1.0]])/sqrt(2)
def g(x):
    return PlaneWaves(direction, k).values(x)
def gn(x,n):
    return PlaneWaves(direction, k).derivs(x,n)

bnddata={11:zero_dirichlet(),
         10:generic_boundary_data([-1j*k,1],[-1j*k,1],g=g,dg=gn)}

runParallel()

bounds=array([[-2,2],[-2,2],[0,0]],dtype='d')
npoints=array([200,200,1])

comp.writeSolution(bounds,npoints,fname='soundsoft.vti')
comp.writeMesh(fname='soundsoft.vtu',scalars=comp.combinedError())
