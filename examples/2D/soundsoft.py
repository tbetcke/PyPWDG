from pypwdg.core.bases import PlaneWaves
from pypwdg import setup,runParallel,gmshMesh
from pypwdg import zero_dirichlet,generic_boundary_data
from numpy import array,sqrt

k = 15
def g(x):
    return PlaneWaves(array([1,0])/sqrt(2), k).values(x)
def gn(x,n):
    return PlaneWaves(array([1,0]),k).derivs(x,n)

bnddata={11:zero_dirichlet(),
         10:generic_boundary_data(array([-1j*k,1]),array([-1j*k,1]),g,gn)}

runParallel()

#bnddata={11:zero_dirichlet(), 
#         10:zero_impedance(k)}

bounds=array([[-2,2],[-2,2],[0,0]],dtype='d')
npoints=array([200,200,1])

comp=setup(gmshMesh('squarescatt.msh',dim=2),k=10,nquadpoints=40,nplanewaves=30,bnddata=bnddata)
comp.writeSolution(bounds,npoints,fname='soundsoft.vti')
comp.writeMesh(fname='soundsoft.vtu',scalars=comp.combinedError())